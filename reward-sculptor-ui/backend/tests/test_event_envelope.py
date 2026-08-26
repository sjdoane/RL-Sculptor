"""The stdout event envelope must not overwrite payload fields.

Every `[SCULPT-EVENT]` line the sculpt subprocess prints gets a provenance
marker stamped on it. That marker used to be written to `source` — a key the
emitters already use for real data (a warm start's checkpoint path, a
selection's origin, a clip's dataset), so all of it arrived at the UI reading
"stdout". Caught when the Training tab's "Started from" card rendered a
checkpoint path as the literal word `stdout`.
"""
from __future__ import annotations

import asyncio
import json

from backend.services.run_manager import EVENT_TAG, _stream_stdout


class _FakeStdout:
    def __init__(self, lines: list[str]):
        self._lines = [ln.encode() for ln in lines]

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class _FakeProc:
    def __init__(self, lines: list[str]):
        self.stdout = _FakeStdout(lines)


class _FakeJob:
    def __init__(self):
        self.events: list[dict] = []

    def emit(self, ev: dict) -> None:
        self.events.append(ev)


def _pump(lines: list[str], tmp_path) -> list[dict]:
    job = _FakeJob()
    asyncio.run(_stream_stdout(
        job=job, proc=_FakeProc(lines), log_path=tmp_path / "run.log",
    ))
    return [e for e in job.events if e.get("type") != "log_line"]


def test_a_payloads_own_source_survives_the_envelope(tmp_path):
    ckpt = "/projects/p/runs/iter_1/checkpoint.pt"
    (ev,) = _pump([
        EVENT_TAG + " " + json.dumps({
            "type": "warm_start_loaded", "source": ckpt,
            "source_sha8": "21c1495a"}) + "\n",
    ], tmp_path)

    assert ev["source"] == ckpt, "the checkpoint path must reach the UI intact"
    assert ev["origin"] == "stdout", "provenance still recorded, under its own key"
    assert ev["source_sha8"] == "21c1495a"


def test_an_event_with_no_source_still_reports_stdout(tmp_path):
    """Back-compat for consumers reading `source` as provenance."""
    (ev,) = _pump([
        EVENT_TAG + " " + json.dumps({"type": "iter_started", "iter": 4}) + "\n",
    ], tmp_path)

    assert ev["source"] == "stdout"
    assert ev["origin"] == "stdout"


def test_every_line_is_still_logged_and_non_event_lines_pass_through(tmp_path):
    job = _FakeJob()
    asyncio.run(_stream_stdout(
        job=job,
        proc=_FakeProc(["plain progress line\n",
                        EVENT_TAG + ' {"type": "run_started"}\n',
                        "not json: " + EVENT_TAG + " {oops\n"]),
        log_path=tmp_path / "run.log",
    ))

    logged = [e for e in job.events if e["type"] == "log_line"]
    assert [e["text"] for e in logged] == [
        "plain progress line",
        EVENT_TAG + ' {"type": "run_started"}',
        "not json: " + EVENT_TAG + " {oops",
    ]
    typed = [e for e in job.events if e["type"] != "log_line"]
    assert [e["type"] for e in typed] == ["run_started"]
    assert (tmp_path / "run.log").read_text().splitlines()[0] == "plain progress line"


def test_parsed_event_is_forwarded_to_observation_boundary(tmp_path):
    job = _FakeJob()
    observed: list[dict] = []
    asyncio.run(_stream_stdout(
        job=job,
        proc=_FakeProc([
            EVENT_TAG + " " + json.dumps({
                "type": "warm_start_loaded",
                "source": "/project/runs/iter_0/checkpoint.pt",
                "source_sha8": "12345678",
                "load_cfg_keys": ["actor"],
            }) + "\n",
        ]),
        log_path=tmp_path / "run.log",
        on_event=observed.append,
    ))

    assert len(observed) == 1
    assert observed[0]["type"] == "warm_start_loaded"
    assert observed[0]["origin"] == "stdout"
    assert observed[0]["source"].endswith("checkpoint.pt")
