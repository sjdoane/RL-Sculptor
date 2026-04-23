"""Tests for GET/PUT /projects/{slug}/rewards.

The PUT path exercises the subprocess-isolated compute_reward probe +
AST validation + parent-hash concurrency check without invoking the
LLM.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_project(client: TestClient, name: str = "Rewards") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 10, "behavior_goal": "run"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _parent_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _human_source(parent_path: Path, *, alive_bonus: float = 2.5) -> str:
    return textwrap.dedent(f'''\
        """v1 — human-authored by test."""
        from __future__ import annotations


        REWARD_SPEC: dict = {{
            "version": "v1",
            "parent_hash": "{_parent_hash(parent_path)}",
            "description": "Bumped alive_bonus by hand.",
            "author": "human",
            "hyperparameters": {{
                "alive_bonus": {alive_bonus},
            }},
            "references": [],
        }}


        def compute_reward(state, action, next_state, info):
            alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
            components = {{"alive_bonus": alive}}
            return alive, components
        ''')


# ── list + detail ─────────────────────────────────────────────────────
def test_list_rewards_fresh_project_shows_v0(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/rewards")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["version"] == 0
    assert body[0]["file_name"] == "v0.py"
    assert body[0]["author"] == "human"  # sculpt init stamps human


def test_get_reward_detail_returns_source_and_probe(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/rewards/0")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 0
    assert "def compute_reward" in body["source"]
    assert body["spec"]["version"] == "v0"
    assert body["components_probe"]["ok"] is True
    # sculpt_init's stub returns {"alive_bonus": 1.0}, total = 1.0
    assert body["components_probe"]["components"] == {"alive_bonus": 1.0}
    assert body["components_probe"]["total"] == pytest.approx(1.0)


def test_get_missing_version_404(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/rewards/99")
    assert r.status_code == 404


# ── PUT manual edit ──────────────────────────────────────────────────
def test_manual_edit_creates_v1(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    assert v0_path.is_file()

    body = {
        "source": _human_source(v0_path, alive_bonus=2.5),
        "expected_parent_version": 0,
        "note": "bump alive bonus",
    }
    r = client.put(f"/projects/{slug}/rewards/0", json=body)
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["version"] == 1
    assert detail["author"] == "human"
    assert detail["components_probe"]["ok"] is True
    assert detail["components_probe"]["components"]["alive_bonus"] == pytest.approx(2.5)

    # Filesystem: v1.py exists + current.py re-exports it.
    assert (tmp_projects_root / slug / "rewards" / "v1.py").is_file()
    current = (tmp_projects_root / slug / "rewards" / "current.py").read_text("utf-8")
    assert "v1.py" in current

    # List now has two versions, newest first.
    r = client.get(f"/projects/{slug}/rewards")
    assert r.status_code == 200
    vs = r.json()
    assert [v["version"] for v in vs] == [1, 0]


def test_manual_edit_rejects_wrong_parent_version_409(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    body = {
        "source": _human_source(v0_path),
        "expected_parent_version": 42,  # nonsense
    }
    r = client.put(f"/projects/{slug}/rewards/42", json=body)
    assert r.status_code == 409
    problem = r.json()
    assert problem["type"] == "/problems/concurrency-conflict"
    assert problem.get("current") == 0


def test_manual_edit_path_vs_body_mismatch_422(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    body = {
        "source": _human_source(v0_path),
        "expected_parent_version": 1,  # disagrees with URL 0
    }
    r = client.put(f"/projects/{slug}/rewards/0", json=body)
    assert r.status_code == 422


def test_manual_edit_rejects_non_human_author(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    source = _human_source(v0_path).replace('"author": "human"', '"author": "sculptor"')
    r = client.put(
        f"/projects/{slug}/rewards/0",
        json={"source": source, "expected_parent_version": 0},
    )
    assert r.status_code == 422
    problem = r.json()
    assert problem["type"] == "/problems/reward-validation"
    assert any("author" in v.lower() for v in problem["violations"])


def test_manual_edit_rejects_syntax_error(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    r = client.put(
        f"/projects/{slug}/rewards/0",
        json={
            "source": "this is not( python",
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 422
    problem = r.json()
    assert problem["type"] == "/problems/reward-validation"


def test_manual_edit_rejects_bad_signature(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    # Wrong signature — missing `info`.
    source = _human_source(v0_path).replace(
        "def compute_reward(state, action, next_state, info):",
        "def compute_reward(state, action, next_state):",
    )
    r = client.put(
        f"/projects/{slug}/rewards/0",
        json={"source": source, "expected_parent_version": 0},
    )
    assert r.status_code == 422
    problem = r.json()
    assert problem["type"] == "/problems/reward-validation"
    assert any("compute_reward" in v for v in problem["violations"])


def test_manual_edit_rejects_bad_parent_hash(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    source = _human_source(v0_path).replace(
        _parent_hash(v0_path), "deadbeef" + "00" * 4
    )
    r = client.put(
        f"/projects/{slug}/rewards/0",
        json={"source": source, "expected_parent_version": 0},
    )
    assert r.status_code == 422
    problem = r.json()
    assert problem["type"] == "/problems/reward-validation"
    assert any("parent_hash" in v for v in problem["violations"])


def test_manual_edit_blocked_when_sculpt_run_active(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slug = _make_project(client)
    # Simulate a running sculpt_run in the JobManager. We reach in via
    # the app state — the TestClient fixture exposes the app.
    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager

    from backend.services.job_manager import Job
    jm._jobs["job_fake_run"] = Job(
        job_id="job_fake_run",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )

    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    r = client.put(
        f"/projects/{slug}/rewards/0",
        json={
            "source": _human_source(v0_path),
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 409
    problem = r.json()
    assert problem["type"] == "/problems/state-conflict"
    assert "sculpt run" in (problem.get("detail") or "").lower()


# ── POST regenerate-template ─────────────────────────────────────────
def _swap_config_adapter_to_mjlab(
    project_root: Path, slug: str,
) -> None:
    """Flip the [adapter].class in config.toml to the mjlab dotted path,
    simulating a broken-state project that was scaffolded under one
    adapter and has its class pointer updated post-hoc (the exact state
    every pre-fix mjlab project ended up in)."""
    cfg_path = project_root / slug / "config.toml"
    text = cfg_path.read_text(encoding="utf-8")
    new = text.replace(
        "sculptor.adapters.gym_sb3.GymSB3Adapter",
        "sculptor.adapters.mjlab.MjlabAdapter",
    )
    assert new != text, "expected gym_sb3 class in the default scaffold"
    cfg_path.write_text(new, encoding="utf-8")


def test_regenerate_template_flips_scalar_to_batched_for_mjlab(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Happy path: project currently has the scalar v0.py (gym_sb3
    scaffold). After swapping [adapter].class to mjlab and calling
    regenerate, v0.py now exports compute_reward_batched."""
    slug = _make_project(client)
    v0_path = tmp_projects_root / slug / "rewards" / "v0.py"
    assert "compute_reward_batched" not in v0_path.read_text(encoding="utf-8")

    _swap_config_adapter_to_mjlab(tmp_projects_root, slug)

    r = client.post(f"/projects/{slug}/rewards/regenerate-template")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["version"] == 0
    assert "def compute_reward_batched" in detail["source"]
    # Filesystem mirrors the response.
    assert "def compute_reward_batched(" in v0_path.read_text(encoding="utf-8")


def test_regenerate_template_noop_shape_for_gym_sb3(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """For a gym_sb3 project (the default), regenerate rewrites v0 with
    the scalar template — shape-unchanged."""
    slug = _make_project(client)
    r = client.post(f"/projects/{slug}/rewards/regenerate-template")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["version"] == 0
    assert "def compute_reward(" in detail["source"]
    assert "def compute_reward_batched" not in detail["source"]


def test_regenerate_template_404_on_unknown_slug(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.post("/projects/does-not-exist/rewards/regenerate-template")
    assert r.status_code == 404
    assert r.json()["type"] == "/problems/not-found"


def test_regenerate_template_409_when_sculpt_run_active(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager

    from backend.services.job_manager import Job
    jm._jobs["job_fake_run"] = Job(
        job_id="job_fake_run",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )

    r = client.post(f"/projects/{slug}/rewards/regenerate-template")
    assert r.status_code == 409
    problem = r.json()
    assert problem["type"] == "/problems/state-conflict"
    assert "sculpt run" in (problem.get("detail") or "").lower()


def test_regenerate_template_preserves_current_py_when_v1_exists(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """If the project has iterated past v0, current.py must keep
    re-exporting the latest (not regress to v0)."""
    slug = _make_project(client)
    _swap_config_adapter_to_mjlab(tmp_projects_root, slug)

    rewards_dir = tmp_projects_root / slug / "rewards"
    # Simulate a v1 that was manually iterated past v0.
    (rewards_dir / "v1.py").write_text(
        "REWARD_SPEC = {}\ndef compute_reward(s,a,n,i): return 9.0, {}\n",
        encoding="utf-8",
    )
    (rewards_dir / "current.py").write_text(
        "from .v1 import *  # noqa: F401,F403\n",
        encoding="utf-8",
    )

    r = client.post(f"/projects/{slug}/rewards/regenerate-template")
    assert r.status_code == 200
    # v0 got rewritten with the batched template.
    v0_src = (rewards_dir / "v0.py").read_text(encoding="utf-8")
    assert "def compute_reward_batched(" in v0_src
    # current.py still points at v1.
    current = (rewards_dir / "current.py").read_text(encoding="utf-8")
    assert "v1" in current


# ── S6 (§7.3 / T7): grounding dict surfaced on reward detail ──────
def test_reward_detail_surfaces_grounding_dict(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A v<n>.py that declares `REWARD_SPEC["grounding"] = {...}` (per
    the 2026-04-22 KG-grounding mandate) must surface that dict on
    `GET /projects/{slug}/rewards/{version}.spec.grounding`. Pre-S6
    the backend serializer silently dropped the field, so the UI had
    nothing to render in the new Grounding column."""
    slug = _make_project(client)
    rewards_dir = tmp_projects_root / slug / "rewards"
    v0_path = rewards_dir / "v0.py"
    parent_hash = hashlib.sha256(v0_path.read_bytes()).hexdigest()[:16]
    v1_path = rewards_dir / "v1.py"
    v1_path.write_text(
        textwrap.dedent(f'''\
            """v1 — with grounding dict."""
            from __future__ import annotations

            REWARD_SPEC: dict = {{
                "version": "v1",
                "parent_hash": "{parent_hash}",
                "author": "sculptor",
                "description": "Added alive bonus with citation.",
                "hyperparameters": {{
                    "alive_bonus": 2.5,
                }},
                "references": [
                    {{
                        "arxiv_id": "1707.02286",
                        "citation": "Mnih et al. 2017 survey",
                        "how_used": "Baseline alive_bonus value",
                    }},
                ],
                "grounding": {{
                    "alive_bonus": "1707.02286 — Mnih survey recommends 1-3 range",
                    "ctrl_cost_weight": "physics-first-principles: normalize by torque bound",
                }},
            }}


            def compute_reward(state, action, next_state, info):
                alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
                return alive, {{"alive_bonus": alive}}
            '''),
        encoding="utf-8",
    )

    r = client.get(f"/projects/{slug}/rewards/1")
    assert r.status_code == 200, r.text
    body = r.json()
    spec = body["spec"]
    assert "grounding" in spec
    assert spec["grounding"]["alive_bonus"].startswith("1707.02286")
    assert "physics-first-principles" in spec["grounding"]["ctrl_cost_weight"]


def test_reward_detail_grounding_defaults_empty_for_pre_mandate_rewards(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Pre-mandate v0.py has no `grounding` key. The serializer must
    default to `{}` so the UI can distinguish 'no mandate applied' from
    a buggy-missing-field rendering path."""
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/rewards/0")
    assert r.status_code == 200
    spec = r.json()["spec"]
    assert spec.get("grounding") == {}


# ── Test 1 follow-up (Issue A): schema-aware probe ─────────────────
def test_probe_components_handles_dict_state_for_schema_contracts(
    tmp_path: Path,
) -> None:
    """Regression for 'AttributeError: float object has no attribute
    items' on Cartpole v1 probe: when a state_schema is supplied,
    probe_components must build nested-list dicts so Claude's
    `state["qpos"]` etc. don't crash inside the subprocess.
    """
    from backend.services.reward_store import probe_components

    reward_path = tmp_path / "v1.py"
    reward_path.write_text(
        textwrap.dedent('''\
            """v1 — schema-aware reward (mjlab-style)."""
            REWARD_SPEC = {
                "version": "v1",
                "parent_hash": "x" * 16,
                "hyperparameters": {"w_upright": 1.0},
                "references": [],
                "grounding": {"w_upright": "physics: upright bonus"},
            }
            def compute_reward(state, action, next_state, info):
                # Reads state as a dict — would crash if probe passed 0.0.
                qpos = state["qpos"]
                cart = float(qpos[0][0])
                pole = float(qpos[0][1])
                upright = 1.0 - pole * pole
                return upright, {"upright": upright, "cart": -cart * cart}
            '''),
        encoding="utf-8",
    )
    probe = probe_components(
        reward_path,
        state_schema={"qpos": (2,), "qvel": (2,), "actuator_force": (1,)},
        info_keys=["episode_length", "terminated"],
    )
    assert probe.ok, probe.error
    assert set(probe.components.keys()) == {"upright", "cart"}


def test_probe_components_scalar_mode_still_works_for_gym_sb3(
    tmp_path: Path,
) -> None:
    """The scalar (no-schema) path must still work for gym_sb3 rewards
    that treat state/action/next_state as plain floats. Guards against
    a regression where the schema branch broke the default."""
    from backend.services.reward_store import probe_components

    reward_path = tmp_path / "v0_gym.py"
    reward_path.write_text(
        textwrap.dedent('''\
            """v0 — scalar gym-style reward."""
            REWARD_SPEC = {
                "version": "v0", "parent_hash": "", "author": "sculptor",
                "hyperparameters": {"alive_bonus": 1.0},
                "references": [],
            }
            def compute_reward(state, action, next_state, info):
                return 1.0, {"alive_bonus": 1.0}
            '''),
        encoding="utf-8",
    )
    probe = probe_components(reward_path)  # No schema passed.
    assert probe.ok, probe.error
    assert probe.components == {"alive_bonus": 1.0}
    assert probe.total == pytest.approx(1.0)
