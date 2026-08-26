"""The editor must never be asked to retype the immutable reference tables.

A 6.92 s composed clip produced a 19752-byte tracking base, 51 % of it dense
float tables the model is contractually forbidden from changing. The resulting
17956-token ceiling truncated attempt 1; the truncation reminder then told the
model not to restate unchanged code, attempt 2 obeyed, and the run died with
`immutable reference tracking kernel changed` before iteration 1.

These cover the fix: elide the tables before the call, splice them back before
validation. Nothing about the kernel-hash or SHA256 gates is relaxed — the
tables are now identical by construction rather than merely checked.
"""

from __future__ import annotations

import textwrap

from sculptor.edit import (
    _ELIDED_TABLE_SENTINEL,
    _elide_reference_tables,
    _reference_kernel_hash,
    _restore_reference_tables,
    _rewrite_token_ceiling,
)


def _module(*, joint_pos_rows: int = 200, residual: str = "return 0.0") -> str:
    """A reference-tracking module shaped like the real generated one."""
    rows = ",\n    ".join(
        f"[{0.001 * i:.6f}, {-0.002 * i:.6f}, {0.003 * i:.6f}]"
        for i in range(joint_pos_rows)
    )
    # NB: built without textwrap.dedent — dedent would strip only the rows'
    # 4-space indent and leave the module body indented, so nothing would be
    # at top level for the AST walk to find.
    return _TEMPLATE.format(rows=rows, residual=residual)


_TEMPLATE = '''\
"""Generated tracking-residual reward."""
REFERENCE_TARGET_SHA256 = "abc123"
REFERENCE_N_JOINTS = 3
REFERENCE_JOINT_POS = [
    {rows}
]
_W_JOINT_POS = 1.0
_W_JOINT_VEL = 0.1
_W_ROOT = 0.5
_W_ORIENTATION = 0.25
_TRACKING_WEIGHT = 1.0
_RESIDUAL_MAX = 0.2
_ALIVE_BONUS = 0.05


def _scalar(x):
    return float(x)


def reference_clock_scalar(info):
    return 0.0


def _phase_index_scalar(t):
    return 0


def _reference_tracking_numpy(obs):
    return 0.0


def reference_clock_batched(info, like):
    return like


def reference_target_index_batched(info, like):
    return like


def _phase_index_batched(t):
    return 0


def _reference_tracking_batched(obs):
    return 0.0


def _task_residual(obs):
    {residual}


def compute_reward(obs):
    return _reference_tracking_numpy(obs) + _task_residual(obs)


def compute_reward_batched(obs):
    return _reference_tracking_batched(obs)
'''


def test_elision_removes_the_table_and_keeps_the_module_parseable():
    src = _module()
    redacted, blocks = _elide_reference_tables(src)

    assert "REFERENCE_JOINT_POS" in blocks
    assert len(redacted) < len(src) / 2, "the table dominates the module"
    assert f"REFERENCE_JOINT_POS = {_ELIDED_TABLE_SENTINEL}" in redacted
    # The model reads this as Python, so it has to parse.
    compile(redacted, "<redacted>", "exec")


def test_small_reference_scalars_are_left_inline():
    """Eliding a 24-byte assignment costs more attention than it saves."""
    _redacted, blocks = _elide_reference_tables(_module())
    assert "REFERENCE_N_JOINTS" not in blocks
    assert "REFERENCE_TARGET_SHA256" not in blocks


def test_unchanged_round_trip_is_byte_identical():
    src = _module()
    redacted, blocks = _elide_reference_tables(src)
    assert _restore_reference_tables(redacted, blocks) == src


def test_restore_survives_an_edit_that_shifts_line_numbers():
    """The realistic case: the model adds a residual above and below the table."""
    src = _module()
    redacted, blocks = _elide_reference_tables(src)
    edited = redacted.replace(
        "def _task_residual(obs):\n    return 0.0",
        "def _task_residual(obs):\n"
        "    # a longer residual\n"
        "    forward = obs.get('vel_x', 0.0)\n"
        "    return min(_RESIDUAL_MAX, 0.1 * forward)",
    ).replace(
        '"""Generated tracking-residual reward."""',
        '"""Generated tracking-residual reward.\n\nEdited: forward progress.\n"""',
    )
    assert edited != redacted, "fixture must actually edit the module"

    restored = _restore_reference_tables(edited, blocks)
    assert blocks["REFERENCE_JOINT_POS"][2] in restored
    assert "forward = obs.get('vel_x', 0.0)" in restored
    assert _ELIDED_TABLE_SENTINEL not in restored
    compile(restored, "<restored>", "exec")


def test_restored_module_keeps_the_parent_kernel_hash():
    """The whole point: an edit to the residual alone must still validate."""
    src = _module()
    redacted, blocks = _elide_reference_tables(src)
    edited = redacted.replace("return 0.0\n\n\ndef compute_reward",
                              "return 0.25\n\n\ndef compute_reward")
    restored = _restore_reference_tables(edited, blocks)
    assert _reference_kernel_hash(restored) == _reference_kernel_hash(src)


def test_a_dropped_table_is_left_absent_for_the_kernel_gate_to_catch():
    """Silently re-appending would paper over a real structural edit."""
    src = _module()
    redacted, blocks = _elide_reference_tables(src)
    dropped = "\n".join(
        line for line in redacted.splitlines()
        if not line.startswith("REFERENCE_JOINT_POS")
    ) + "\n"

    restored = _restore_reference_tables(dropped, blocks)
    assert "REFERENCE_JOINT_POS" not in restored
    assert _reference_kernel_hash(restored) != _reference_kernel_hash(src)


def test_elision_shrinks_the_output_ceiling_the_model_must_fit():
    src = _module(joint_pos_rows=400)
    redacted, _blocks = _elide_reference_tables(src)
    # Not an equality assertion on the constant — the point is that the budget
    # is now sized to what the model emits, not to data it never sees.
    assert _rewrite_token_ceiling(redacted) <= _rewrite_token_ceiling(src)
    assert len(redacted) < len(src) / 3


def test_a_non_reference_module_is_untouched():
    plain = textwrap.dedent('''\
        """An ordinary reward."""
        WEIGHTS = {"alive": 1.0}


        def compute_reward(obs):
            return WEIGHTS["alive"]
    ''')
    redacted, blocks = _elide_reference_tables(plain)
    assert blocks == {}
    assert redacted == plain
    assert _restore_reference_tables(redacted, blocks) == plain


def test_syntactically_broken_output_is_returned_unchanged():
    """A truncated response must reach the existing SyntaxError path intact."""
    src = _module()
    _redacted, blocks = _elide_reference_tables(src)
    truncated = "def compute_reward(obs):\n    return ("
    assert _restore_reference_tables(truncated, blocks) == truncated


def test_truncation_reminder_never_tells_the_model_to_omit_code():
    """The regression that killed the platform-ascent run."""
    import inspect

    from sculptor import edit as edit_mod

    src = inspect.getsource(edit_mod)
    # Anchor on the raise, not the first textual match — the module docstring
    # and the NB comment above the raise both quote this phrase.
    raise_at = src.index("raise EditValidationError(")
    start = src.index("response was cut off at the", raise_at)
    message = src[start:start + 600]
    assert "do not restate unchanged" not in message
    assert "must still" in message and "present" in message
