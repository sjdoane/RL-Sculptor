"""§Ship 31: pre-campaign grounding hardening tests.

F1 — prompt-feeding semantic queries are floored (diagnose/decompose
     now share edit's 0.35 min-similarity).
F2 — hallucinated paper_refs are dropped AT DIAGNOSE TIME (observable),
     never riding into edit's hard-fail gate.
F3 — failure-mode resolution is scored + deterministic (the old
     fallback took the FIRST node matching ANY single token).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sculptor.kg.query import (
    DEFAULT_MIN_PROMPT_SIMILARITY,
    _resolve_failure_modes,
)
from sculptor.kg.schema import (
    FailureMode,
    make_failure_mode_id,
)
from sculptor.kg.store import SculptorKG


# ── F3: failure-mode resolution ──────────────────────────────────────


@pytest.fixture()
def fm_store(tmp_path: Path) -> SculptorKG:
    kg = SculptorKG(tmp_path / "kg.db")
    for name, desc in [
        ("reward_saturation",
         "a reward component sits at its maximum and provides no gradient"),
        ("reward_hacking",
         "the policy exploits the reward without doing the behavior"),
        ("sparse_reward",
         "reward signal too infrequent for learning progress"),
        ("foot_slip", "feet slide on the ground during stance"),
        ("early_termination", "episodes end before the behavior matures"),
    ]:
        kg.add_node(FailureMode(
            id=make_failure_mode_id(name), name=name, description=desc,
        ))
    return kg


def test_resolve_exact_slug_wins(fm_store: SculptorKG) -> None:
    out = _resolve_failure_modes(fm_store, ["reward_saturation"])
    assert out["reward_saturation"].name == "reward_saturation"


def test_resolve_prefers_full_phrase_over_token_soup(fm_store: SculptorKG) -> None:
    """'reward saturation problem' contains the full phrase 'reward
    saturation' → must resolve to reward_saturation, not whichever
    node happens to contain the token 'reward' first."""
    out = _resolve_failure_modes(fm_store, ["reward saturation problem"])
    assert out["reward saturation problem"].name == "reward_saturation"


def test_resolve_majority_token_requirement(fm_store: SculptorKG) -> None:
    """A multi-token target matching only ONE of its tokens must NOT
    resolve (the old any-single-token bug grounded it arbitrarily)."""
    out = _resolve_failure_modes(
        fm_store, ["zebra wormhole reward dance party"],
    )
    # 1 of 5 tokens ('reward') < majority → unresolved.
    assert "zebra wormhole reward dance party" not in out


def test_resolve_is_deterministic(fm_store: SculptorKG) -> None:
    outs = [
        _resolve_failure_modes(fm_store, ["policy exploits reward"])
        for _ in range(3)
    ]
    names = {o["policy exploits reward"].name for o in outs if o}
    assert len(names) == 1
    assert names == {"reward_hacking"}


# ── F1: floored prompt queries ───────────────────────────────────────


def test_floor_constant_matches_edit() -> None:
    from sculptor.edit import apply_prompt_edit  # noqa: F401 — import side check
    import sculptor.edit as edit_mod

    src = Path(edit_mod.__file__).read_text(encoding="utf-8")
    assert "_MIN_PROMPT_EDIT_SIMILARITY = 0.35" in src
    assert DEFAULT_MIN_PROMPT_SIMILARITY == 0.35


def test_diagnose_and_decompose_pass_floor() -> None:
    """Source-level pin: the two call sites that feed prompts must pass
    the shared floor (a regression here silently reopens Issue G)."""
    import sculptor.diagnose as diag_mod
    import sculptor.decompose as dec_mod

    diag_src = Path(diag_mod.__file__).read_text(encoding="utf-8")
    dec_src = Path(dec_mod.__file__).read_text(encoding="utf-8")
    assert "min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY" in diag_src
    assert "min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY" in dec_src


def test_query_semantic_floor_filters(tmp_path: Path) -> None:
    """Functional check of the floor on a tiny store: a query unrelated
    to the lone technique must return nothing at 0.35."""
    from sculptor.kg.schema import Technique, make_technique_id
    from sculptor.kg.query import query_semantic

    kg = SculptorKG(tmp_path / "kg.db")
    kg.add_node(Technique(
        id=make_technique_id("foot_clearance_reward"),
        name="foot_clearance_reward",
        description="reward swing-foot apex height to avoid scuffing",
    ))
    hits_related = query_semantic(
        "feet drag on the ground during swing", store=kg, top_k=3,
        min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY,
    )
    hits_unrelated = query_semantic(
        "stock market portfolio rebalancing strategies", store=kg, top_k=3,
        min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY,
    )
    assert len(hits_related) == 1
    assert hits_unrelated == []


# ── F2: citation verification at diagnose time ───────────────────────


def test_diagnose_drops_hallucinated_refs(tmp_path: Path, capsys, monkeypatch) -> None:
    """A grounded response citing an arxiv_id absent from the KG must
    have that ref DROPPED (edit degrades to novel) with an observable
    kg_citation_dropped event — not ride into edit's hard-fail gate."""
    import sculptor.diagnose as diag

    from sculptor.kg.schema import Paper, make_paper_id

    kg = SculptorKG(tmp_path / "kg.db")
    kg.add_node(Paper(
        id=make_paper_id("1707.06347"),
        arxiv_id="1707.06347", title="PPO", year=2017,
    ))

    prelim = diag._PreliminaryModel(
        failure_modes=["reward_saturation"],
        evidence="component pinned at max",
        confidence=0.8,
    )
    grounded = diag._GroundedModel(
        proposed_edits=[
            diag._ProposedEditModel(
                target_term="alive_bonus",
                operation="increase",
                rationale="cited one real and one fabricated paper",
                suggested_value="0.5",
                paper_refs=["1707.06347", "9999.99999"],
            ),
        ],
        confidence=0.7,
    )

    class _StubMessages:
        def __init__(self):
            self._payloads = [prelim, grounded]

        def parse(self, **kwargs):
            class R:
                def __init__(self, p):
                    self.parsed_output = p
            return R(self._payloads.pop(0))

    class _StubClient:
        def __init__(self):
            self.messages = _StubMessages()

    iter_dir = tmp_path / "iter_0"
    (iter_dir / "rollout").mkdir(parents=True)
    (iter_dir / "rollout" / "behavior.json").write_text(json.dumps({
        "mean_return": 1.0, "mean_episode_length": 100.0,
        "max_episode_length": 500, "n_episodes": 2,
    }))
    (iter_dir / "metrics.json").write_text(json.dumps({"mean_return": 1.0}))
    config = tmp_path / "config.toml"
    config.write_text(
        '[adapter]\n'
        'class = "tests.test_eval_harness._EvalStubAdapter"\n'
        'config = { task_id = "T" }\n'
        '[kg]\n'
        'environment_tag = "test"\n'
    )

    d = diag.diagnose(
        iter_dir,
        "balance",
        config,
        client=_StubClient(),
        store=kg,
    )
    refs = d.proposed_edits[0].paper_refs
    assert refs == ["1707.06347"], refs
    out = capsys.readouterr().out
    assert "kg_citation_dropped" in out
    assert "9999.99999" in out
