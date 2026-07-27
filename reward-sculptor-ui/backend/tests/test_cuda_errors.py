"""Tests for backend/services/cuda_errors.classify (M4 §5)."""

from __future__ import annotations

from backend.services.cuda_errors import classify


def test_oom_detected_and_suggests_smaller_num_envs() -> None:
    msg = (
        "RuntimeError: torch.cuda.OutOfMemoryError: CUDA out of memory. "
        "Tried to allocate 2.00 GiB"
    )
    r = classify(msg, current_num_envs=4096)
    assert r.kind == "oom"
    assert r.suggested_num_envs is not None
    assert r.suggested_num_envs <= 2048
    assert "num_envs" in r.suggestions[0] or "num_envs" in r.detail


def test_oom_variants() -> None:
    for text in (
        "CUDNN error: CUDNN_STATUS_ALLOC_FAILED",
        "Out Of Memory",
        "torch.cuda.OutOfMemoryError: ...",
    ):
        assert classify(text).kind == "oom", f"miss: {text!r}"


def test_driver_version_detected() -> None:
    msg = "CUDA driver version is insufficient for CUDA runtime version"
    r = classify(msg)
    assert r.kind == "driver_version"
    assert any("driver" in s.lower() or "nvidia" in s.lower() for s in r.suggestions)


def test_no_cuda_detected() -> None:
    msg = "RuntimeError: no CUDA-capable device is detected"
    r = classify(msg)
    assert r.kind == "no_cuda"


def test_unknown_on_generic_error() -> None:
    r = classify("Python exited with SIGSEGV")
    assert r.kind == "unknown"
    assert r.title


def test_policy_distribution_instability_detected() -> None:
    msg = "RuntimeError: normal expects all elements of std >= 0.0"
    r = classify(msg)
    assert r.kind == "policy_distribution_instability"
    assert "exploration" in r.title.lower()
    assert r.problem_type == "/problems/policy-distribution-instability"
    assert any("World" in suggestion for suggestion in r.suggestions)


def test_suggest_num_envs_snaps_to_power_of_two() -> None:
    r = classify("CUDA out of memory", current_num_envs=3000)
    # 3000 // 2 = 1500 — snaps to 1024
    assert r.suggested_num_envs == 1024


def test_suggest_num_envs_floor_128() -> None:
    r = classify("CUDA out of memory", current_num_envs=200)
    assert r.suggested_num_envs == 128


def test_suggest_num_envs_absent_without_hint() -> None:
    r = classify("CUDA out of memory")
    assert r.suggested_num_envs is None


# ── reward-contract mismatch (mjlab v0 scaffolded as scalar) ─────────
def test_reward_contract_mismatch_detected() -> None:
    """Verbatim fragment from `_mjlab_runner.py` when v0.py lacks
    `compute_reward_batched`."""
    msg = (
        "AttributeError: reward module "
        "'/.../rewards/v0.py' missing compute_reward_batched; "
        "required when training with MjlabAdapter (set "
        "REWARD_SPEC['supports_batched']=True ...)"
    )
    r = classify(msg)
    assert r.kind == "reward_contract_mismatch"
    assert "batched" in r.title.lower()
    assert r.problem_type == "/problems/reward-contract-mismatch"
    assert r.action is not None
    assert r.action.get("kind") == "regenerate_reward_template"
    # First suggestion should mention the UI action.
    assert any("Regenerate" in s for s in r.suggestions)


def test_reward_contract_mismatch_wins_over_cuda_mentions() -> None:
    """Reward-contract match must win even when the traceback includes
    CUDA language — we check the contract pattern first."""
    msg = (
        "Traceback ... torch/cuda/__init__.py line 123 ... AttributeError: "
        "reward module missing compute_reward_batched required for MjlabAdapter"
    )
    r = classify(msg)
    assert r.kind == "reward_contract_mismatch"


def test_problem_type_assigned_for_existing_kinds() -> None:
    """All known kinds should set a non-default problem_type so the UI
    can route to the right remediation panel."""
    assert classify("CUDA out of memory").problem_type == "/problems/cuda-oom"
    assert (
        classify("CUDA driver version is insufficient").problem_type
        == "/problems/cuda-driver-too-old"
    )
    assert (
        classify("no CUDA-capable device is detected").problem_type
        == "/problems/cuda-no-device"
    )
    # unknown stays at default sentinel.
    assert classify("SIGSEGV").problem_type == "about:blank"
