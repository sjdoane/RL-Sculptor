"""tests/test_modes_cli.py — the `sculpt modes` surface.

`sculptor.mode_rewards` was a well-tested library with nothing reaching it.
These cover the path a person actually takes: read the automaton out of a
composed clip, scaffold a reward from it, and author one mode at a time.

The authoring call itself needs a model, so what is exercised here is
everything up to and including the prompt — plus the guards that stop a bad
authoring call from starting, which is where the value is: a scaffold that has
drifted from its graph, a mode that does not exist, a missing contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from sculptor.cli import app
from sculptor.mode_rewards import MAX_PROMPT_CHARS

FPS = 120.0
J = 6
SEAMS = [150, 300]
N = 444


def _write_composite(root: Path, clip_id: str = "novel-jump-kick--g1",
                     robot: str = "g1", *, segments=("approach", "launch", "strike"),
                     seams=SEAMS, n=N) -> Path:
    """A composed clip on disk, carrying the composition provenance the
    automaton is derived from. Written directly rather than through
    `refs.compose` so the test stays about the CLI."""
    t = np.arange(n, dtype=np.float64) / FPS
    d = root / robot / clip_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {"clip_id": clip_id,
            "composition": {
                "seam_frames": list(seams),
                "segments": [{"index": i, "label": label, "source_id": f"src_{i}",
                              "source_fps": 60.0, "source_frames": [0, 60]}
                             for i, label in enumerate(segments)]}}
    np.savez(
        d / "clip.npz",
        fps=np.float64(FPS),
        root_pos_z=0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        root_pos_xy=np.stack([0.5 * t, np.zeros(n)], axis=1),
        root_quat_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        joint_pos=0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None] + 0.01 * np.arange(J)[None, :],
        joint_names=np.array([f"joint_{i}" for i in range(J)]),
        meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    return d


def _run(monkeypatch, root: Path, args):
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    return CliRunner().invoke(app, args)


# ── show ────────────────────────────────────────────────────────────────
def test_show_reads_the_automaton_out_of_the_clips_own_provenance(
        tmp_path, monkeypatch):
    """Nothing is invented: one composed segment is one mode, each seam is a
    transition. The windows are seconds, which is what the reward gates on."""
    root = tmp_path / "refs"
    _write_composite(root)
    r = _run(monkeypatch, root, ["modes", "show", "--clip-id", "novel-jump-kick--g1"])
    assert r.exit_code == 0, r.output
    assert "3 modes @ 120 fps" in r.output
    assert "approach: frames [0, 150) = 0.000s–1.250s" in r.output
    assert "strike: frames [300, 444) = 2.500s–3.700s" in r.output
    assert "approach -> launch" in r.output and "launch -> strike" in r.output


def test_show_json_round_trips_the_graph(tmp_path, monkeypatch):
    root = tmp_path / "refs"
    _write_composite(root)
    r = _run(monkeypatch, root,
             ["modes", "show", "--clip-id", "novel-jump-kick--g1", "--json"])
    assert r.exit_code == 0, r.output
    g = json.loads(r.stdout)
    assert [m["name"] for m in g["modes"]] == ["approach", "launch", "strike"]
    assert len(g["transitions"]) == 2


def test_a_single_clip_reference_says_why_it_has_no_automaton(
        tmp_path, monkeypatch):
    """A non-composite is the common mistake, so the message has to explain
    rather than just fail — there is one mode and no transition to derive."""
    root = tmp_path / "refs"
    d = root / "g1" / "plain--g1"
    d.mkdir(parents=True)
    t = np.arange(60, dtype=np.float64) / FPS
    np.savez(d / "clip.npz", fps=np.float64(FPS),
             root_pos_z=0.70 + 0.0 * t,
             joint_pos=np.zeros((60, J)),
             joint_names=np.array([f"joint_{i}" for i in range(J)]))
    r = _run(monkeypatch, root, ["modes", "show", "--clip-id", "plain--g1"])
    assert r.exit_code == 1
    assert "meta.composition" in r.output


def test_a_missing_clip_names_the_path_it_looked_at(tmp_path, monkeypatch):
    r = _run(monkeypatch, tmp_path / "refs",
             ["modes", "show", "--clip-id", "nope--g1"])
    assert r.exit_code == 1 and "nope--g1" in r.output


# ── scaffold ────────────────────────────────────────────────────────────
def test_scaffold_writes_a_module_and_names_the_unauthored_modes(
        tmp_path, monkeypatch):
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "modes" / "v0.py"
    r = _run(monkeypatch, root, [
        "modes", "scaffold", "--clip-id", "novel-jump-kick--g1",
        "--goal", "run in and strike at the apex", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert "3 modes, 3 unauthored" in r.output
    src = out.read_text(encoding="utf-8")
    assert "run in and strike at the apex" in src
    assert src.count("UNAUTHORED STUB") == 6      # scalar + batched per mode
    # The next command is printed for each pending mode — the scaffold is
    # useless without knowing that authoring is the next step.
    for name in ("approach", "launch", "strike"):
        assert f"--mode {name}" in r.output


def test_scaffold_refuses_to_clobber_an_authored_module(tmp_path, monkeypatch):
    """Regenerating discards authored mode bodies, and the scaffold is the
    cheap half — losing the authored terms is the expensive one."""
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    args = ["modes", "scaffold", "--clip-id", "novel-jump-kick--g1",
            "--out", str(out)]
    assert _run(monkeypatch, root, args).exit_code == 0
    again = _run(monkeypatch, root, args)
    assert again.exit_code == 1 and "--force" in again.output
    assert _run(monkeypatch, root, args + ["--force"]).exit_code == 0


def test_scaffold_to_stdout_needs_no_destination(tmp_path, monkeypatch):
    root = tmp_path / "refs"
    _write_composite(root)
    r = _run(monkeypatch, root,
             ["modes", "scaffold", "--clip-id", "novel-jump-kick--g1"])
    assert r.exit_code == 0
    assert "def compute_reward_batched(" in r.stdout
    assert "MODE_WINDOWS_S" in r.stdout


# ── author ──────────────────────────────────────────────────────────────
def test_print_prompt_states_the_window_and_both_halves(tmp_path, monkeypatch):
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--print-prompt",
        "--goal", "run in, launch off one leg, strike at the apex",
        "--mode-goal", "convert horizontal speed into a single-leg takeoff"])
    assert r.exit_code == 0, r.output
    assert "_mode_launch_batched" in r.stdout
    assert "1.25s-2.5s" in r.stdout
    assert "after 'approach'" in r.stdout and "before 'strike'" in r.stdout
    assert len(r.stdout.strip()) <= MAX_PROMPT_CHARS


def test_authoring_without_a_project_explains_what_the_contract_is_for(
        tmp_path, monkeypatch):
    """The contract is not bureaucracy here — it is what the post-flight probe
    checks the authored BATCHED path against before it can reach a GPU."""
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out)])
    assert r.exit_code == 2
    assert "--project is required" in r.output


def test_authoring_an_unknown_mode_lists_the_real_ones(tmp_path, monkeypatch):
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "landing", "--file", str(out), "--print-prompt"])
    assert r.exit_code == 1
    assert "approach, launch, strike" in r.output


def test_authoring_into_a_scaffold_that_drifted_from_its_graph_is_refused(
        tmp_path, monkeypatch):
    """The dangerous one. Authoring terms into a window that has since moved
    produces a module that trains happily and rewards the wrong slice of the
    motion, so it is caught before the model is ever called."""
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])

    # Re-cut the same clip's seams: same modes, different windows.
    _write_composite(root, seams=[120, 260])
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--print-prompt"])
    assert r.exit_code == 1
    assert "stale" in r.output and "regenerate" in r.output.lower()


def test_authoring_before_scaffolding_says_which_command_to_run(
        tmp_path, monkeypatch):
    root = tmp_path / "refs"
    _write_composite(root)
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(tmp_path / "nothing.py"),
        "--print-prompt"])
    assert r.exit_code == 1 and "modes scaffold" in r.output


def _fake_adapter(monkeypatch, project: Path):
    """A project whose adapter yields an mjlab-shaped contract, without mjlab.

    The point is to exercise everything `modes author` does AROUND the model
    call — build the twin, graft, write, re-probe — which is where the code
    under test lives.
    """
    import types

    project.mkdir(parents=True, exist_ok=True)
    (project / "config.toml").write_text("[target]\nname = 'fake'\n")
    contract = types.SimpleNamespace(
        supports_batched=True,
        state_schema={"qpos": (29,), "projected_gravity_b": (3,),
                      "actuator_force": (29,)},
        info_schema={"episode_length": (), "step_dt": (), "base_height": ()},
        expected_info_keys=["episode_length", "step_dt", "base_height"])
    monkeypatch.setattr(
        "sculptor.adapters.base.load_adapter",
        lambda _p: types.SimpleNamespace(reward_contract=lambda: contract))
    return contract


def test_authoring_builds_a_twin_grafts_it_back_and_re_probes(
        tmp_path, monkeypatch):
    """The whole `modes author` machinery minus the model.

    This is the path that had a `NameError: name 'clip' is not defined` in it,
    invisible to every other test here because `--print-prompt` returns before
    reaching it."""
    from sculptor.mode_rewards import MODE_FN_PREFIX, authored_modes

    root = tmp_path / "refs"
    _write_composite(root, seams=[150, 300], n=444)
    out = tmp_path / "v0.py"
    goal = "run in and strike at the apex"
    assert _run(monkeypatch, root, [
        "modes", "scaffold", "--clip-id", "novel-jump-kick--g1",
        "--goal", goal, "--out", str(out)]).exit_code == 0
    assert "TARGET_JOINT_POS" in out.read_text(), "backbone is on by default"

    project = tmp_path / "proj"
    _fake_adapter(monkeypatch, project)

    seen = {}

    def _fake_edit(*, current_reward_path, user_prompt, new_iter_id, **_kw):
        # What the model would return: the twin with 'launch' filled in.
        src = Path(current_reward_path).read_text(encoding="utf-8")
        seen["twin"] = src
        seen["prompt"] = user_prompt
        for fn, body in (
            (f"{MODE_FN_PREFIX}launch(state, action, next_state, info)",
             "    del state, action, next_state, info\n    return 0.25, {'takeoff': 0.25}\n"),
            (f"{MODE_FN_PREFIX}launch_batched(state, action, next_state, info, like)",
             "    del state, action, next_state, info\n    return like + 0.25, {'takeoff': like + 0.25}\n"),
        ):
            head = f"def {fn}:"
            i = src.index(head)
            j = src.index("\ndef ", i)
            src = src[:i] + head + "\n" + body + src[j:]
        dest = Path(current_reward_path).parent / f"{new_iter_id}.py"
        dest.write_text(src, encoding="utf-8")
        return dest

    monkeypatch.setattr("sculptor.edit.apply_prompt_edit", _fake_edit)
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--project", str(project),
        "--goal", goal])
    assert r.exit_code == 0, r.output

    # The twin the model saw: no float tables, but state-dependent.
    # The literals are the whole reason the twin exists — a real g1 scaffold
    # is 32x29 of them and the model mangled the table when asked to
    # reproduce it. Size alone understates this here (the fixture has 6
    # joints), so assert on the literals themselves.
    scaffold = out.read_text()
    floats = re.compile(r"-?\d+\.\d{3,}")

    def _table_rows(src):  # a row of the target table, vs. receipt/window bounds
        return [
            ln
            for ln in src.splitlines()
            if ln.lstrip().startswith("[") and len(floats.findall(ln)) >= 3
        ]

    assert len(_table_rows(scaffold)) > 10
    assert _table_rows(seen["twin"]) == []
    assert "np.zeros((N_PHASE, N_JOINTS)" in seen["twin"]
    assert len(seen["twin"]) < len(scaffold)
    assert "_mode_launch_batched" in seen["prompt"]

    # And the grafted result carries the authored mode AND the real tables.
    v1 = (tmp_path / "v1.py").read_text(encoding="utf-8")
    assert authored_modes(v1)["launch"] is True
    assert authored_modes(v1)["approach"] is False
    assert "np.zeros((N_PHASE, N_JOINTS)" not in v1, "real targets, not the twin's"
    assert "0.25" in v1
    assert "1/3 modes authored" in r.output
    assert "--mode approach" in r.output and "--mode strike" in r.output


def test_a_helper_the_model_defines_is_carried_across_the_graft(
        tmp_path, monkeypatch):
    """The second real authoring run died here.

    The model wrote the mode plus an `_info_b` helper at module level; the
    graft took only the two mode functions, and the result failed the batched
    probe with `NameError: name '_info_b' is not defined`."""
    from sculptor.mode_rewards import MODE_FN_PREFIX

    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    project = tmp_path / "proj"
    _fake_adapter(monkeypatch, project)

    def _edit_with_helper(*, current_reward_path, new_iter_id, **_kw):
        src = Path(current_reward_path).read_text(encoding="utf-8")
        # A helper calling a second helper, to pin the transitive case.
        src += ("\n\ndef _launch_scale():\n    return _launch_gain() * 2.0\n"
                "\n\ndef _launch_gain():\n    return 0.125\n")
        for fn, body in (
            (f"{MODE_FN_PREFIX}launch(state, action, next_state, info)",
             "    del state, action, next_state, info\n"
             "    v = _launch_scale()\n    return v, {'takeoff': v}\n"),
            (f"{MODE_FN_PREFIX}launch_batched(state, action, next_state, info, like)",
             "    del state, action, next_state, info\n"
             "    v = like + _launch_scale()\n    return v, {'takeoff': v}\n"),
        ):
            head = f"def {fn}:"
            i = src.index(head)
            j = src.index("\ndef ", i)
            src = src[:i] + head + "\n" + body + src[j:]
        dest = Path(current_reward_path).parent / f"{new_iter_id}.py"
        dest.write_text(src, encoding="utf-8")
        return dest

    monkeypatch.setattr("sculptor.edit.apply_prompt_edit", _edit_with_helper)
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--project", str(project)])
    assert r.exit_code == 0, r.output

    v1 = (tmp_path / "v1.py").read_text(encoding="utf-8")
    assert "def _launch_scale()" in v1 and "def _launch_gain()" in v1
    # Carried once, not once per grafted function.
    assert v1.count("def _launch_scale()") == 1
    ns: dict = {}
    exec(compile(v1, "v1.py", "exec"), ns)
    assert ns["_mode_launch"]({}, None, {"qpos": [0.0] * 6}, {})[0] == 0.25


def test_the_twin_does_not_grow_as_modes_get_authored(tmp_path, monkeypatch):
    """The third call has to fit in the same budget as the first.

    Authoring is one mode per call and `apply_prompt_edit` regenerates the
    whole module, so carrying each finished mode's body into the next twin
    made it grow monotonically — mode 'approach' hit that live and both
    attempts came back truncated. The twin carries what a neighbour PAYS, not
    how."""
    from sculptor.mode_rewards import MODE_FN_PREFIX

    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    project = tmp_path / "proj"
    _fake_adapter(monkeypatch, project)

    twins = {}

    def _edit_for(mode):
        # A deliberately bulky body, so a twin that carried it would show.
        filler = "\n".join(f"    x{i} = {i}.0 + float(len(info))" for i in range(60))

        def _edit(*, current_reward_path, new_iter_id, **_kw):
            src = Path(current_reward_path).read_text(encoding="utf-8")
            twins[mode] = src
            for suffix, sig, ret in (
                ("", "(state, action, next_state, info)", "0.5"),
                ("_batched", "(state, action, next_state, info, like)", "like + 0.5"),
            ):
                fn = f"{MODE_FN_PREFIX}{mode}{suffix}"
                head = f"def {fn}{sig}:"
                i = src.index(head)
                j = src.index("\ndef ", i)
                src = (src[:i] + head + "\n    del state, action, next_state\n"
                       + filler + f"\n    v = {ret}\n"
                       + f"    return v, {{'{mode}_core': v, '{mode}_aux': v}}\n"
                       + src[j:])
            dest = Path(current_reward_path).parent / f"{new_iter_id}.py"
            dest.write_text(src, encoding="utf-8")
            return dest
        return _edit

    src_file = out
    for n, mode in enumerate(("approach", "launch", "strike"), start=1):
        monkeypatch.setattr("sculptor.edit.apply_prompt_edit", _edit_for(mode))
        r = _run(monkeypatch, root, [
            "modes", "author", "--clip-id", "novel-jump-kick--g1",
            "--mode", mode, "--file", str(src_file), "--project", str(project)])
        assert r.exit_code == 0, r.output
        assert f"{n}/3 modes authored" in r.output
        src_file = tmp_path / f"v{n}.py"

    # The third twin is no bigger than the first, despite two finished modes.
    assert len(twins["strike"]) <= len(twins["approach"]) * 1.1
    # And it says what they pay without carrying how.
    assert "pays approach_core, approach_aux" in twins["strike"]
    assert "x59 = 59.0" not in twins["strike"], "neighbour bodies stayed out"
    # while the real module has every one of them.
    final = src_file.read_text(encoding="utf-8")
    assert final.count("x59 = 59.0") == 6, "3 modes x scalar+batched"


def _author_reading(monkeypatch, keys):
    """Patch the edit call to author `launch` reading `keys` out of info."""
    from sculptor.mode_rewards import MODE_FN_PREFIX

    reads = " + ".join(f"float(info.get({k!r}, 0.0))" for k in keys) or "0.0"
    treads = " + ".join(f"info.get({k!r}, like)" for k in keys) or "like"

    def _edit(*, current_reward_path, new_iter_id, **_kw):
        src = Path(current_reward_path).read_text(encoding="utf-8")
        for fn, body in (
            (f"{MODE_FN_PREFIX}launch(state, action, next_state, info)",
             f"    del state, action, next_state\n"
             f"    v = {reads}\n    return v, {{'takeoff': v}}\n"),
            (f"{MODE_FN_PREFIX}launch_batched(state, action, next_state, info, like)",
             f"    del state, action, next_state\n"
             f"    v = like + ({treads})\n    return v, {{'takeoff': v}}\n"),
        ):
            head = f"def {fn}:"
            i = src.index(head)
            j = src.index("\ndef ", i)
            src = src[:i] + head + "\n" + body + src[j:]
        dest = Path(current_reward_path).parent / f"{new_iter_id}.py"
        dest.write_text(src, encoding="utf-8")
        return dest

    monkeypatch.setattr("sculptor.edit.apply_prompt_edit", _edit)


def test_a_mode_reading_an_info_key_the_env_never_sends_is_rejected(
        tmp_path, monkeypatch):
    """The failure no other probe can see.

    `info.get(key, 0.0)` on a key the env does not publish imports fine, runs
    fine, validates fine — and pays a constant for the whole of training. That
    is the exact shape of a gameable reward, so it is a rejection."""
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    project = tmp_path / "proj"
    _fake_adapter(monkeypatch, project)
    _author_reading(monkeypatch, ["base_height", "toe_pressure_left"])

    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--project", str(project)])
    assert r.exit_code == 1
    assert "toe_pressure_left" in r.output
    assert "base_height" in r.output, "the real key is listed as available"
    assert "constant" in r.output


def test_a_mode_reading_only_declared_keys_passes_the_key_gate(
        tmp_path, monkeypatch):
    """The gate has to stay quiet on the generated backbone, which reads
    `base_height_delta`/`base_height` itself — a false positive here would
    make every authored mode unauthorable."""
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    project = tmp_path / "proj"
    _fake_adapter(monkeypatch, project)
    _author_reading(monkeypatch, ["base_height", "episode_length"])

    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--project", str(project)])
    assert r.exit_code == 0, r.output
    assert "1/3 modes authored" in r.output


def test_a_model_that_edits_nothing_is_reported_rather_than_accepted(
        tmp_path, monkeypatch):
    """A silent no-op is worse than a rejection: the next call would move on to
    the following mode, leaving this one an unpaid stub."""
    root = tmp_path / "refs"
    _write_composite(root)
    out = tmp_path / "v0.py"
    _run(monkeypatch, root, ["modes", "scaffold", "--clip-id",
                             "novel-jump-kick--g1", "--out", str(out)])
    project = tmp_path / "proj"
    _fake_adapter(monkeypatch, project)

    def _no_op(*, current_reward_path, new_iter_id, **_kw):
        dest = Path(current_reward_path).parent / f"{new_iter_id}.py"
        dest.write_text(Path(current_reward_path).read_text(), encoding="utf-8")
        return dest

    monkeypatch.setattr("sculptor.edit.apply_prompt_edit", _no_op)
    r = _run(monkeypatch, root, [
        "modes", "author", "--clip-id", "novel-jump-kick--g1",
        "--mode", "launch", "--file", str(out), "--project", str(project)])
    assert r.exit_code == 1
    assert "unauthored stub" in r.output


def test_the_authored_output_name_chains_across_modes():
    """Authoring is one mode per call, so the versions have to chain — and a
    hand-placed scaffold with no version to bump must not overwrite itself on
    the second call."""
    from sculptor.cli import _next_iter_id

    assert _next_iter_id("v0", "launch") == "v1"
    assert _next_iter_id("v9", "launch") == "v10"
    assert _next_iter_id("mode_reward", "launch") == "mode_reward_launch"
    assert _next_iter_id("mode_reward", "one leg") == "mode_reward_one_leg"
