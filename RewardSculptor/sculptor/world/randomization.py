"""Per-episode domain randomization for authored worlds.

The world author already declares a `train.variations` list (box heights, gap
lengths, object mass/friction) with sampling distributions, but until now it was
compiled into metadata and never consumed — every episode trained against ONE
frozen layout, which is exactly what undercuts sim-to-real transfer.

This module turns those declared variations into mjlab reset-time events so each
parallel environment re-samples its layout every episode. Two paths:

* **Object entities** (a ball / cube in an object task) — mass and friction are
  randomized with mjlab's own proven entity DR (`dr.body_mass` / `dr.geom_friction`),
  which handle per-env model-field expansion and indexing.
* **Course platforms** (parkour boxes) — the authored course is raw worldbody
  geometry (not an mjlab entity), so a small custom reset event
  (`randomize_authored_course`) resolves the box geoms by name and re-samples each
  platform's HEIGHT per env, moving `geom_size.z` and `geom_pos.z` together so the
  box grows from the ground, then recomputes the broadphase bounds (mirroring
  mjlab's own `dr.geom_size`).

Everything is TRAIN-ONLY (evaluation always replays the frozen manifest) and
fail-safe: a variation that cannot be resolved is logged and skipped, and the
reset event never raises (a crash there would kill training).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

try:
    # Marks which model fields mjlab must expand PER-WORLD before a reset can
    # write them (without this, geom_size/geom_pos stay a single shared array and
    # the per-env write is a no-op / error). Same mechanism mjlab's own dr.geom_*
    # funcs use. A no-op fallback keeps this module importable without mjlab (the
    # pure parse/sample paths + their tests never need it).
    from mjlab.managers.event_manager import requires_model_fields
except Exception:  # noqa: BLE001
    def requires_model_fields(*fields: str, **_kw: Any):  # type: ignore[misc]
        def _deco(fn):
            fn.model_fields = tuple(fields)
            return fn
        return _deco

# JSON-pointer grammars for the variation targets the author emits.
_COURSE_RE = re.compile(
    r"^/shared/obstacles/course/@(?P<id>[^/]+)/nominal/(?P<field>[a-z_]+)$")
_OBJECT_RE = re.compile(
    r"^/shared/objects/(?P<name>[^/]+)/nominal/(?P<field>[a-z_]+)$")
_OBJECT_POSITION_RE = re.compile(
    r"^/shared/objects/(?P<name>[^/]+)/nominal/pose/position_m/(?P<axis>[012])$")


@dataclass(frozen=True)
class Randomization:
    """One resolved per-episode randomization: WHAT to perturb and its range."""
    var_id: str
    kind: str                     # "course_height" | "object_mass" | "object_friction"
    target_name: str              # geom name (course) or entity name (object)
    low: float
    high: float
    axis: int | None = None


def _range_from_distribution(dist: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    """Collapse a declared distribution to a (low, high) sampling range.

    `uniform` is exact; `normal` uses mean ± 2σ (clipped when given); `choice`
    spans its values. Anything unrecognized / degenerate returns None (skip)."""
    kind = dist.get("kind")
    try:
        if kind == "uniform":
            lo, hi = float(dist["low"]), float(dist["high"])
        elif kind == "normal":
            mean, std = float(dist["mean"]), float(dist.get("std", 0.0))
            lo, hi = mean - 2.0 * std, mean + 2.0 * std
            clip = dist.get("clip")
            if isinstance(clip, (list, tuple)) and len(clip) == 2:
                lo, hi = max(lo, float(clip[0])), min(hi, float(clip[1]))
        elif kind == "choice":
            vals = [float(v) for v in dist.get("values", [])]
            if not vals:
                return None
            lo, hi = min(vals), max(vals)
        else:
            return None
    except (TypeError, ValueError, KeyError):
        return None
    if not (hi > lo):
        return None
    return lo, hi


def resolve_world_randomizations(world: Mapping[str, Any]) -> list[Randomization]:
    """Parse `train.variations` into the randomizations this module can install.

    Unrecognized targets/fields are skipped (a gap-length variation, say, is not
    yet wired) — never an error, so a partially-supported world still randomizes
    everything it can."""
    out: list[Randomization] = []
    variations = (world.get("train", {}) or {}).get("variations", []) or []
    for var in variations:
        if not isinstance(var, Mapping):
            continue
        target = str(var.get("target", ""))
        rng = _range_from_distribution(var.get("distribution") or {})
        if rng is None:
            continue
        low, high = rng
        vid = str(var.get("id", target))
        m = _OBJECT_POSITION_RE.match(target)
        if m:
            out.append(Randomization(
                vid, "object_position", m.group("name"), low, high,
                int(m.group("axis"))))
            continue
        m = _COURSE_RE.match(target)
        if m and m.group("field") == "height_m":
            out.append(Randomization(
                vid, "course_height", f"obstacle__{m.group('id')}", low, high))
            continue
        m = _OBJECT_RE.match(target)
        if m:
            field = m.group("field")
            name = m.group("name")
            if field == "mass_kg":
                out.append(Randomization(vid, "object_mass", name, low, high))
            elif field == "friction":
                out.append(Randomization(vid, "object_friction", name, low, high))
    return out


# ── custom reset event: authored-course platform heights ─────────────────────

def _course_geom_ids(env: Any, base_name: str) -> list[int]:
    """All model geom indices for a course element — its base geom AND every
    per-env-origin copy the compiler emits as ``<base>__env_NNNN`` for a
    multi-origin (parallel-training) scene. The authored course is worldbody
    geometry (not an entity), so this scans the CPU mj_model by name.

    Returning ALL copies is what makes course randomization actually fire in
    production: with >1 env origin the base name resolves to nothing, so the old
    single-name lookup was a silent no-op — the exact config the feature exists
    for. Writing the same per-env sample to every copy means whichever origin an
    env sits at, it reads that env's randomized height."""
    try:
        import mujoco

        mj = env.sim.mj_model
        prefix = base_name + "__env_"
        ids: list[int] = []
        for gid in range(int(mj.ngeom)):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name == base_name or (name is not None and name.startswith(prefix)):
                ids.append(gid)
        return ids
    except Exception:  # noqa: BLE001
        return []


def _write_box_heights(model: Any, env_ids: Any, gid: int, heights: Any) -> None:
    """Set a box geom's half-height and center so it grows from the ground, and
    recompute the broadphase bounds — the exact contract `dr.geom_size` upholds.

    Pure tensor math (no name lookup / no device assumptions) so it is unit-
    testable on CPU with a fake model bridge."""
    import torch

    half = heights * 0.5
    model.geom_size[env_ids, gid, 2] = half
    model.geom_pos[env_ids, gid, 2] = half     # bottom stays on the z=0 ground
    # Box broadphase: rbound = |half-extents|, aabb half-size = half-extents.
    sx = model.geom_size[env_ids, gid, 0]
    sy = model.geom_size[env_ids, gid, 1]
    model.geom_rbound[env_ids, gid] = torch.sqrt(sx * sx + sy * sy + half * half)
    model.geom_aabb[env_ids, gid, 1, 2] = half


def _sample_uniform(low: float, high: float, n: int, device: Any) -> Any:
    import torch

    return low + (high - low) * torch.rand(n, device=device)


@requires_model_fields("geom_size", "geom_pos", "geom_rbound", "geom_aabb")
def randomize_authored_course(env: Any, env_ids: Any, *, specs: Sequence[Mapping[str, Any]]) -> None:
    """mjlab reset event: re-sample each authored course platform's height per env.

    `specs` is a list of `{geom_name, low, high}`. The `@requires_model_fields`
    decorator tells mjlab to expand geom_size/geom_pos/broadphase per-world so
    the per-env writes below take effect. Fail-safe by contract — any unresolved
    geom is skipped and the whole event is wrapped so a reset can never raise."""
    try:
        import torch

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = env_ids.to(env.device)
        model = env.sim.model
        for spec in specs:
            gids = _course_geom_ids(env, str(spec["geom_name"]))
            if not gids:
                continue
            # One sample per env, applied to EVERY per-origin copy of this box.
            heights = _sample_uniform(
                float(spec["low"]), float(spec["high"]),
                int(env_ids.shape[0]), env.device)
            for gid in gids:
                _write_box_heights(model, env_ids, gid, heights)
    except Exception as e:  # noqa: BLE001 — a reset event must never crash a run
        print(f"[world-dr] course randomization skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)


# ── install into an mjlab env_cfg (train only) ───────────────────────────────

def install_world_randomizations(env_cfg: Any, world: Mapping[str, Any]) -> list[str]:
    """Add reset-mode events to `env_cfg.events` so authored `train.variations`
    actually vary the scene per episode. Returns a human-readable list of what was
    installed (for the runner log). No-op (empty) when there is nothing to wire or
    the env has no `events` dict — never raises."""
    rands = resolve_world_randomizations(world)
    if not rands:
        return []
    events = getattr(env_cfg, "events", None)
    if not isinstance(events, dict):
        return []
    try:
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.envs.mdp import dr
    except Exception as e:  # noqa: BLE001 — headless / mjlab missing
        print(f"[world-dr] mjlab DR API unavailable, skipping: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return []

    installed: list[str] = []
    course_specs: list[dict[str, Any]] = []
    for r in rands:
        if r.kind == "course_height":
            course_specs.append(
                {"geom_name": r.target_name, "low": r.low, "high": r.high})
        elif r.kind == "object_mass":
            events[f"world_dr__{r.var_id}"] = EventTermCfg(
                mode="reset", func=dr.body_mass,
                params={
                    "asset_cfg": SceneEntityCfg(
                        r.target_name, body_names=(r.target_name,)),
                    "operation": "abs", "ranges": (r.low, r.high),
                })
            installed.append(f"{r.var_id}: object '{r.target_name}' mass "
                             f"∈[{r.low:.3g},{r.high:.3g}]kg per reset")
        elif r.kind == "object_friction":
            # Regex over ALL of the object's geoms — a sphere/box object has one
            # ("<name>_geom"), but a `frame` (goal) has several
            # ("<name>_left_post", "<name>_crossbar", …); `<name>_.*` covers both
            # (SceneEntityCfg geom_names are regex, matched full).
            events[f"world_dr__{r.var_id}"] = EventTermCfg(
                mode="reset", func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg(
                        r.target_name, geom_names=(f"{r.target_name}_.*",)),
                    "operation": "abs", "ranges": (r.low, r.high),
                })
            installed.append(f"{r.var_id}: object '{r.target_name}' friction "
                             f"∈[{r.low:.3g},{r.high:.3g}] per reset")
        elif r.kind == "object_position" and r.axis is not None:
            events[f"world_dr__{r.var_id}"] = EventTermCfg(
                mode="reset", func=dr.body_pos,
                params={
                    "asset_cfg": SceneEntityCfg(
                        r.target_name, body_names=(r.target_name,)),
                    "operation": "abs", "ranges": (r.low, r.high),
                    "axes": [r.axis],
                })
            axis_name = "xyz"[r.axis]
            installed.append(
                f"{r.var_id}: object '{r.target_name}' {axis_name} "
                f"∈[{r.low:.3g},{r.high:.3g}]m per reset")

    if course_specs:
        events["world_dr__course"] = EventTermCfg(
            mode="reset", func=randomize_authored_course,
            params={"specs": tuple(course_specs)})
        heights = ", ".join(
            f"{s['geom_name']}∈[{s['low']:.3g},{s['high']:.3g}]m"
            for s in course_specs)
        installed.append(f"course platform heights per reset: {heights}")
    return installed
