"""§REFERENCE_TRAJECTORY_PLAN §6: reference-derived calibration ladders —
steer-rights for a NOVEL motion by ranking a competence ladder built FROM a
single attached reference clip. Fully offline/deterministic: no LLM call,
no network dependency.

§audit-finding close (REFERENCE_BUILD_LOG.md "Audit findings deferred" —
Tier-D spoofing) + §F5 follow-up: `calibrate_metric_against_reference` no
longer takes a caller-supplied `tier: str`, NOR a caller-supplied
`tierd_cert: TierDCertificate` (an earlier revision's narrower but still
caller-controlled bypass). There is now NO trust input the caller can hand
in at all — the effective tier is ALWAYS derived by calling
`sculptor.refs.track.verify_tierd_certificate(robot, clip_id,
root=library_root)` internally, which re-derives the verdict from disk on
every call. Tests that need Tier-D rights build a REAL, verified
certificate on disk first (`_seed_real_tierd_cert`, mirroring
`sculptor.refs.track.track_clip`'s own on-disk artifact chain) rather than
constructing/injecting one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sculptor.eval.metric_calibration import (
    calibrate_metric_against_reference,
    compute_trust,
)
from sculptor.reference import save_clip
from sculptor.refs import library
from sculptor.refs.track import (
    TrackingErrors,
    build_tierd_execution_contract,
    update_provenance_tier_d,
)

# ── fixtures ────────────────────────────────────────────────────────────

#: A clean monotone-rising synthetic clip: root height ramps from 0.55 m to
#: 1.05 m with a smooth joint oscillation riding along. The height range is
#: kept ABOVE the fixed battery's degenerate anchor (still/chaotic at
#: z=0.5, fallen at z=0.1 -> anchor 0.5 for a pure-height metric) so an
#: HONEST metric's ladder orders correctly rung-by-rung, including rung 0.
_T = 150
_FPS = 50.0


def _rising_clip() -> dict:
    z = np.linspace(0.55, 1.05, _T)
    t = np.linspace(0.0, 1.0, _T)
    jp = np.stack([0.3 * np.sin(2 * np.pi * t), 0.2 * np.cos(2 * np.pi * t)], axis=1)
    return {
        "root_pos_z": z.astype(np.float64),
        "fps": _FPS,
        "joint_pos": jp,
        "joint_names": ["j0", "j1"],
    }


def _execution_contract(root: Path, clip: dict, robot: str) -> dict:
    donor = root / "tier_d_donor"
    donor.mkdir(parents=True, exist_ok=True)
    config = donor / "config.toml"
    config.write_text(
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Velocity-Flat-Unitree-G1" }\n',
        encoding="utf-8",
    )
    joints = list(clip["joint_names"])
    policy_contract = {
        "identity": {
            "adapter_class": "sculptor.adapters.mjlab.MjlabAdapter",
            "task_id": "Mjlab-Velocity-Flat-Unitree-G1",
        },
        "joints": {"ordered_names": joints},
        "actions": {
            "ordered_names": joints,
            "term_names": ["joint_position"],
            "shape": [len(joints)],
        },
        "timing": {
            "sim_timestep_s": 0.005,
            "decimation": 4,
            "control_dt_s": 0.02,
        },
        "versions": {
            "torch": "2.7",
            "mjlab": "0.3.1",
            "rsl_rl": "3.1.0",
            "adapter": "0.7.0",
        },
    }
    return build_tierd_execution_contract(
        donor_project=donor,
        certification_config_path=config,
        robot=robot,
        clip=clip,
        policy_contract=policy_contract,
    )


def _seed_real_tierd_cert(
    root: Path, clip: dict, *, clip_id: str = "rise_clip", robot: str = "g1",
) -> None:
    """Write a REAL, on-disk, internally-consistent Tier-D certificate for
    `(robot, clip_id)` at `root`: clip.npz + provenance with a genuine
    content_sha256 (computed from the written bytes, not a placeholder —
    §F7 hardened `verify_tierd_certificate` recomputes this from disk),
    a rollout artifact INSIDE `root` (§F7 containment check), and
    tracking errors that clear track.py's feasibility thresholds. §F5:
    `calibrate_metric_against_reference` no longer accepts a caller-
    supplied `tierd_cert=`, so a test that needs Tier-D rights must earn
    a real certificate on disk, exactly like a production `track_clip`
    run would, and let `verify_tierd_certificate` resolve it itself."""
    d = library.clip_dir(robot, clip_id, root=root)
    clip_path = d / library.CLIP_FILENAME
    save_clip(clip_path, clip)
    content_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot, source={"kind": "hf_dataset"},
        license="MIT", attribution="x", content_sha256_=content_sha, tier="K")
    library.write_provenance(robot, clip_id, prov, root=root)
    library.rebuild_index(root=root)

    rollout_path = d / "tierD_rollout.npz"
    rollout_path.write_bytes(b"a real tracking rollout")
    errs = TrackingErrors(
        mean_joint_err_rad=0.05, max_joint_err_rad=0.1, root_z_rmse_m=0.02,
        duration_coverage=1.0, common_joint_names=["j0"], n_common_joints=1)
    assert errs.feasible  # sanity: fixture stats are within threshold
    update_provenance_tier_d(
        robot=robot, clip_id=clip_id, errors=errs, iterations=1,
        rollout_path=rollout_path,
        execution_contract=_execution_contract(root, clip, robot),
        root=root)


#: An honest metric: final-decile mean root height, normalized. Monotone in
#: the clip's own construction (degrade damps root height toward the START
#: pose which is LOWER; truncate cuts off the rise early) so it should rank
#: the ladder correctly and clear the firewall (the degenerate battery sits
#: at height 0.5, well under this clip's ramp).
_HONEST_HEIGHT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = z[int(0.9 * n):]
    val = float(np.clip(np.mean(q) / 1.2, 0.0, 1.0))
    return {"spec_score": val}
'''

_CONSTANT = '''def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''

#: The mirror image of the honest metric: LOW final height scores HIGH.
_REVERSED_HEIGHT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = z[int(0.9 * n):]
    val = float(np.clip(1.0 - np.mean(q) / 1.2, 0.0, 1.0))
    return {"spec_score": val}
'''

#: A metric that ranks the height ladder well ENOUGH (0.6 weight) but adds a
#: confound (0.4 weight) rewarding near-zero total joint excursion — the
#: degenerate battery's `still`/`chaotic` archetypes carry EXACTLY zero
#: joint_pos on this clip's own two-joint layout(*), so the confound scores
#: the degenerate anchor competitively with the full reference. This must
#: trip the firewall (rung-0-vs-full-clip margin), independent of rho.
#: (*) the archetype battery is a DIFFERENT 12-joint synthetic body; the
#: confound reads absolute joint_pos magnitude generically, which is still
#: near-zero on the quiet/near-still archetypes.
_FIREWALL_GAMEABLE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos")
    root = arrays.get("root_link_pos_w")
    if jp is None or root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    height = float(np.clip(np.mean(z[int(0.9 * n):]) / 1.2, 0.0, 1.0))
    quiet = float(np.clip(1.0 - np.mean(np.abs(jp)) / 0.3, 0.0, 1.0))
    val = float(np.clip(0.6 * height + 0.4 * quiet, 0.0, 1.0))
    return {"spec_score": val}
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


#: An empty, never-populated library root — passing this as `library_root`
#: makes auto-resolution deterministically fail-closed (no clip on disk to
#: find), independent of whatever real reference library happens to exist
#: on the machine running the tests. Bundles `source_kind` too so call
#: sites never need to pass it separately (avoids a duplicate-kwarg
#: TypeError against the spread `**kwargs`).
def _base_kwargs(tmp_path: Path, source_kind: str = "hf_dataset") -> dict:
    return {
        "robot": "g1", "library_root": tmp_path / "empty_lib",
        "source_kind": source_kind,
    }


# ── tests: tier resolution (the audit-finding close) ──────────────────────


def test_honest_metric_grants_steer_with_verified_cert(tmp_path):
    """A REAL on-disk Tier-D certificate (§F5: no caller-supplied
    `tierd_cert=` exists anymore) -> D/steer."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    kwargs = _base_kwargs(tmp_path)
    _seed_real_tierd_cert(kwargs["library_root"], clip)
    res = calibrate_metric_against_reference(p, "rise_clip", clip, **kwargs)

    assert res["ok"] is True
    assert res["method"] == "reference"
    assert res["spearman"] >= 0.7
    assert res["rho_min"] == res["spearman"]
    assert res["rights"] == "steer"
    assert res["trust_tier"] == "reference:D:hf_dataset"
    assert res["clip_id"] == "rise_clip"
    assert res["tier"] == "D"
    assert res["source_kind"] == "hf_dataset"
    assert res["cert_verified"] is True
    assert res["cert_reason"] is None


def test_honest_metric_grants_observe_without_cert(tmp_path):
    """No cert presented and none resolvable on disk -> Tier K -> observe,
    even for an otherwise-honest, otherwise-passing metric."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    assert res["ok"] is True
    assert res["rights"] == "observe"
    assert res["trust_tier"] == "reference:K:hf_dataset"
    assert res["tier"] == "K"
    assert res["cert_verified"] is False
    assert res["cert_reason"] is not None


def test_unrelated_on_disk_cert_does_not_grant_rights_for_this_clip(tmp_path):
    """The exact audit finding, restated for §F5: there is no caller-
    supplied cert parameter left to spoof at all — confirm the auto-
    resolve path is scoped per (robot, clip_id) too. A genuine, verified
    Tier-D certificate for a DIFFERENT clip_id must not vouch for THIS
    one — the metric still only earns observe, never steer."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    kwargs = _base_kwargs(tmp_path)
    _seed_real_tierd_cert(kwargs["library_root"], clip, clip_id="some_other_clip")
    res = calibrate_metric_against_reference(p, "rise_clip", clip, **kwargs)

    assert res["ok"] is True
    assert res["rights"] == "observe"
    assert res["tier"] == "K"
    assert res["cert_verified"] is False


def test_calibration_auto_resolves_tierd_cert_from_disk(tmp_path):
    """A REAL, verifiable Tier-D certificate sits on disk for this exact
    clip_id/robot (`_seed_real_tierd_cert`, the same on-disk artifact
    chain `sculptor.refs.track.track_clip` produces) —
    `calibrate_metric_against_reference` must find it itself via
    `verify_tierd_certificate` (there is no other way in — §F5) and
    grant steer."""
    root = tmp_path / "lib"
    clip = _rising_clip()
    _seed_real_tierd_cert(root, clip)

    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, robot="g1", source_kind="hf_dataset",
        library_root=root)

    assert res["rights"] == "steer"
    assert res["tier"] == "D"
    assert res["cert_verified"] is True


def test_calibration_auto_resolve_denies_untracked_clip(tmp_path):
    """A clip that was never tracked (no tierD block at all) auto-resolves
    to K, never D, even though the clip genuinely exists in the library."""
    root = tmp_path / "lib"
    clip = _rising_clip()
    d = library.clip_dir("g1", "rise_clip", root=root)
    from sculptor.reference import save_clip

    save_clip(d / library.CLIP_FILENAME, clip)
    prov = library.make_provenance(
        clip_id="rise_clip", robot="g1", source={"kind": "hf_dataset"},
        license="MIT", attribution="x", content_sha256_="c" * 64, tier="K")
    library.write_provenance("g1", "rise_clip", prov, root=root)
    library.rebuild_index(root=root)

    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, robot="g1", source_kind="hf_dataset",
        library_root=root)

    assert res["rights"] == "observe"
    assert res["tier"] == "K"
    assert res["cert_verified"] is False


# ── tests: ladder mechanics (unchanged by the tier-resolution rewrite) ────


def test_ladder_order_recorded_correctly(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    ladder = res["ladder"]
    names = [r["rung"] for r in ladder]
    ranks = [r["intended_rank"] for r in ladder]
    assert names == [
        "degenerate_battery", "degrade_heavy", "degrade_light",
        "truncate_25", "truncate_50", "truncate_75", "full_clip",
    ]
    assert ranks == [0, 1, 2, 3, 4, 5, 6]
    # Every rung recorded a finite numeric score.
    for rung in ladder:
        assert np.isfinite(rung["score"])


def test_constant_metric_denied(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "const.py", _CONSTANT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    assert res["ok"] is False
    assert res["rights"] == "none"
    assert res["spearman"] == 0.0


def test_reversed_metric_denied(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "reversed.py", _REVERSED_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    assert res["ok"] is False
    assert res["rights"] == "none"
    assert res["spearman"] < 0.0


def test_firewall_denies_degenerate_battery_gaming(tmp_path):
    """A metric that ranks the ladder somewhat but scores the fixed battery's
    degenerate anchor competitively with the full reference must be denied
    by the firewall — even though it may show SOME rank correlation."""
    clip = _rising_clip()
    p = _write(tmp_path, "gameable.py", _FIREWALL_GAMEABLE)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    assert res["ok"] is False
    assert res["rights"] == "none"
    ladder = {r["rung"]: r["score"] for r in res["ladder"]}
    # The firewall margin is violated: full_clip does not clear
    # degenerate_battery by the required spread.
    assert ladder["full_clip"] < ladder["degenerate_battery"] + 0.1
    assert "degenerate anchor" in res["reason"]


def test_trust_tier_string_exact(tmp_path):
    """§F5: tier can no longer be toggled per-call via a caller-supplied
    cert — it is purely a function of on-disk state. Use a fresh library
    root per case, seeding a real Tier-D cert only where D is expected."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    for i, (want_d, source_kind, expect_tier) in enumerate((
        (True, "hf_dataset", "D"),
        (False, "video", "K"),
        (True, "generated", "D"),
    )):
        root = tmp_path / f"lib_{i}"
        if want_d:
            _seed_real_tierd_cert(root, clip)
        res = calibrate_metric_against_reference(
            p, "rise_clip", clip, robot="g1", library_root=root,
            source_kind=source_kind)
        assert res["trust_tier"] == f"reference:{expect_tier}:{source_kind}"


def test_rights_none_when_not_ok(tmp_path):
    """Rights are "none" whenever `ok` is False, REGARDLESS of tier — a
    Tier-D-eligible clip still earns nothing from a denied (constant)
    metric."""
    clip = _rising_clip()
    p = _write(tmp_path, "const.py", _CONSTANT)
    for i, want_d in enumerate((True, False)):
        root = tmp_path / f"lib_{i}"
        if want_d:
            _seed_real_tierd_cert(root, clip)
        res = calibrate_metric_against_reference(
            p, "rise_clip", clip, robot="g1", library_root=root,
            source_kind="hf_dataset")
        assert res["rights"] == "none"


def test_compute_trust_consumes_reference_result_without_error(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    trust = compute_trust(res)
    assert "trust" in trust
    assert 0.0 <= trust["trust"] <= 1.0
    assert trust["method"] == "reference"
    assert trust["rho_min"] == res["rho_min"]
    assert trust["agreement_fraction"] == res["agreement_fraction"]


def test_compute_trust_consumes_denied_reference_result_without_error(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "const.py", _CONSTANT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    trust = compute_trust(res)
    assert "trust" in trust
    assert 0.0 <= trust["trust"] <= 1.0


def test_load_failure_never_raises(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("this is not ( valid python", encoding="utf-8")
    clip = _rising_clip()
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    assert res["ok"] is False
    assert res["rights"] == "none"
    assert res["error"] is not None


def test_cert_resolution_never_raises_on_corrupt_library_root(tmp_path):
    """A `library_root` that points at garbage (not even a directory) must
    not crash calibration — cert resolution fails closed to K/observe."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    garbage_root = tmp_path / "not_a_dir"
    garbage_root.write_text("nope", encoding="utf-8")

    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, robot="g1", source_kind="hf_dataset",
        library_root=garbage_root)

    assert res["tier"] == "K"
    assert res["cert_verified"] is False


def test_determinism(tmp_path):
    """Everything is seeded — two calls on the same inputs must produce byte-
    identical ladders (degrade()'s seed=0 default)."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res1 = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))
    res2 = calibrate_metric_against_reference(
        p, "rise_clip", clip, **_base_kwargs(tmp_path))

    assert res1["ladder"] == res2["ladder"]
    assert res1["spearman"] == res2["spearman"]
