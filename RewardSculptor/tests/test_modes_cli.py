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


def test_the_authored_output_name_chains_across_modes():
    """Authoring is one mode per call, so the versions have to chain — and a
    hand-placed scaffold with no version to bump must not overwrite itself on
    the second call."""
    from sculptor.cli import _next_iter_id

    assert _next_iter_id("v0", "launch") == "v1"
    assert _next_iter_id("v9", "launch") == "v10"
    assert _next_iter_id("mode_reward", "launch") == "mode_reward_launch"
    assert _next_iter_id("mode_reward", "one leg") == "mode_reward_one_leg"
