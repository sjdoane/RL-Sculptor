"""A promoted per-mode reward must survive diagnose-and-edit intact.

`edit.py` gates the reference-tracking kernel on
`REWARD_SPEC["composition"]["type"]`, which a promoted mode reward does not
have; its arrays are `TARGET_*` not `REFERENCE_*` and its kernels are named
`_phase_index`/`_tracking`, so `_reference_kernel_hash` returns None and every
tracking guard switches off. The rewriter could therefore drop a mode, move a
window, or collapse the dispatch to one flat function and pass every gate —
the mode whose terms stopped being paid would just never be paid again.

These cover the mode-structure guard. It is deliberately narrow: the automaton
is frozen, every `_mode_*` body stays fully editable.
"""

from __future__ import annotations

import types

import pytest

from sculptor.edit import (
    EditValidationError,
    _elide_reference_tables,
    _mode_reward_contract,
    _validate_mode_reward_contract,
)


def _mod(*, order=("approach", "bound", "settle"),
         windows=None, fns=None) -> types.SimpleNamespace:
    if windows is None:
        windows = {"approach": (0.0, 0.82), "bound": (0.82, 2.12),
                   "settle": (2.12, 6.92)}
    names = list(fns if fns is not None else order)
    table = {n: (lambda *a, **k: (0.0, {})) for n in names}
    return types.SimpleNamespace(
        MODE_ORDER=list(order),
        MODE_WINDOWS_S=dict(windows),
        _MODE_FNS=dict(table),
        _MODE_FNS_BATCHED=dict(table),
        compute_reward=lambda *a, **k: 0.0,
        compute_reward_batched=lambda *a, **k: 0.0,
    )


def _parent(**kw) -> dict:
    contract = _mode_reward_contract(_mod(**kw))
    assert contract is not None
    return contract


def test_a_flat_reward_has_no_mode_contract():
    """The old single-function path must be untouched by any of this."""
    flat = types.SimpleNamespace(REWARD_SPEC={"version": "v3"},
                                 compute_reward=lambda *a: 0.0)
    assert _mode_reward_contract(flat) is None


def test_contract_captures_order_and_windows():
    contract = _parent()
    assert contract["order"] == ["approach", "bound", "settle"]
    assert contract["windows"]["bound"] == [0.82, 2.12]


def test_an_unchanged_rewrite_passes():
    _validate_mode_reward_contract(mod=_mod(), source="", parent=_parent())


def test_editing_a_mode_body_is_still_allowed():
    """The whole point of running the loop on a per-mode reward."""
    edited = _mod()
    edited._MODE_FNS["bound"] = lambda *a, **k: (1.0, {"height": 1.0})
    _validate_mode_reward_contract(mod=edited, source="", parent=_parent())


def test_dropping_a_mode_is_rejected():
    with pytest.raises(EditValidationError, match="MODE_ORDER changed"):
        _validate_mode_reward_contract(
            mod=_mod(order=("approach", "settle"),
                     windows={"approach": (0.0, 0.82), "settle": (2.12, 6.92)}),
            source="", parent=_parent())


def test_reordering_modes_is_rejected():
    with pytest.raises(EditValidationError, match="MODE_ORDER changed"):
        _validate_mode_reward_contract(
            mod=_mod(order=("bound", "approach", "settle")),
            source="", parent=_parent())


def test_moving_a_window_is_rejected():
    """Windows come from the composition seams — moving one desyncs the clip."""
    moved = _mod(windows={"approach": (0.0, 2.0), "bound": (2.0, 2.12),
                          "settle": (2.12, 6.92)})
    with pytest.raises(EditValidationError, match=r"moved from"):
        _validate_mode_reward_contract(mod=moved, source="", parent=_parent())


def test_collapsing_the_dispatch_to_a_flat_function_is_rejected():
    flat = types.SimpleNamespace(compute_reward=lambda *a: 0.0,
                                 compute_reward_batched=lambda *a: 0.0)
    with pytest.raises(EditValidationError, match="structure, not tuning"):
        _validate_mode_reward_contract(mod=flat, source="", parent=_parent())


def test_a_mode_missing_from_the_dispatch_table_is_rejected():
    """MODE_ORDER intact but the body gone — pays nothing or raises."""
    holed = _mod(fns=("approach", "settle"))
    with pytest.raises(EditValidationError, match=r"no entry for mode"):
        _validate_mode_reward_contract(mod=holed, source="", parent=_parent())


def test_losing_the_batched_entry_point_is_rejected():
    """mjlab dispatches to it; its absence is a contract violation."""
    stripped = _mod()
    del stripped.compute_reward_batched
    with pytest.raises(EditValidationError, match="compute_reward_batched"):
        _validate_mode_reward_contract(mod=stripped, source="",
                                       parent=_parent())


def test_mode_reward_target_tables_are_elided_too():
    """`TARGET_*` is the per-mode naming for the same immutable arrays."""
    rows = ",\n    ".join(f"[{0.01 * i:.4f}, {-0.01 * i:.4f}]" for i in range(200))
    src = (f"TARGET_JOINT_POS = [\n    {rows}\n]\n"
           "MODE_ORDER: list = ['approach']\n"
           "def compute_reward(obs):\n    return 0.0\n")
    redacted, blocks = _elide_reference_tables(src)
    assert "TARGET_JOINT_POS" in blocks
    assert len(redacted) < len(src) / 4
    compile(redacted, "<redacted>", "exec")
