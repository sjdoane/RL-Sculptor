"""Classify CUDA / mjlab / sculpt runtime errors into actionable recovery hints.

Used by run-failure pathways: when a training subprocess exits non-zero,
the backend can grep stdout+stderr for known patterns and return a
structured error that the UI renders with a specific remediation.

Patterns match verbatim substrings in CUDA / PyTorch / sculptor error
messages rather than regexes — cheaper and precise enough. Add new
categories incrementally as real failures are observed.

The module is named `cuda_errors` for legacy reasons; it classifies any
subprocess error now, not just CUDA ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CudaErrorClass:
    """Result of classify(text)."""

    # kind: "oom" | "driver_version" | "no_cuda" | "reward_contract_mismatch" | "unknown"
    kind: str
    title: str
    detail: str
    suggestions: list[str]
    # Recovery hint for the UI: a num_envs figure, a link, etc.
    suggested_num_envs: Optional[int] = None
    # Problem-type URI for RFC 7807 responses + UI routing.
    problem_type: str = "about:blank"
    # Optional action descriptor so the UI can render a one-click remediation
    # (e.g. "Regenerate reward template" → calls a known endpoint).
    action: Optional[dict] = None
    # Structured, non-path evidence for stage-aware failures. Kept separate
    # from prose so run/iteration summaries can render truthful progress.
    evidence: Optional[dict] = None


def classify(text: str, *, current_num_envs: Optional[int] = None) -> CudaErrorClass:
    """Scan the subprocess output for known error signatures. Returns an
    `unknown` classification if nothing matches — the UI shows the raw
    stderr in that case.
    """
    needle = (text or "").lower()

    # 1. Reward-contract mismatch — mjlab runner raises AttributeError when
    # the loaded reward module lacks `compute_reward_batched`. Surface
    # BEFORE the generic OOM/driver scanners because the text may include
    # traceback lines that also mention "cuda".
    if "missing compute_reward_batched" in needle:
        return CudaErrorClass(
            kind="reward_contract_mismatch",
            title="Reward template missing batched entry point",
            detail=(
                "The reward module (`rewards/v0.py`) only exports scalar "
                "`compute_reward`, but this project's mjlab runner needs "
                "`compute_reward_batched` for per-step GPU evaluation. "
                "This usually means the project was scaffolded under the "
                "gym_sb3 template and later switched to mjlab."
            ),
            suggestions=[
                "Open the Rewards tab and click \"Regenerate reward template\".",
                "That rewrites rewards/v0.py using the mjlab scaffold "
                "(git history preserves the previous file).",
                "Then relaunch the run.",
            ],
            problem_type="/problems/reward-contract-mismatch",
            action={
                "kind": "regenerate_reward_template",
                "label": "Regenerate reward template",
            },
        )

    # 2. Legacy scalar Gaussian exploration can cross below zero after an
    # optimizer step.  RewardSculptor's mjlab runner installs a positivity
    # guard, but classify older/unpatched runs precisely instead of presenting
    # an empty generic failure card.
    if "normal expects all elements of std >= 0.0" in needle:
        return CudaErrorClass(
            kind="policy_distribution_instability",
            title="Policy exploration became unstable",
            detail=(
                "A directly learned Gaussian action standard deviation crossed "
                "below zero during PPO optimization. The environment and World "
                "build remain valid; the policy distribution needs the scalar-"
                "standard-deviation guard before training can continue."
            ),
            suggestions=[
                "Update RewardSculptor to a build with the policy standard-"
                "deviation guard, then launch a new exact-match run.",
                "Keep the same World selection, seed, environment count, and "
                "training budget so results remain comparable.",
            ],
            problem_type="/problems/policy-distribution-instability",
        )

    # 3. OOM — several wordings in the wild.
    oom_markers = [
        "cuda out of memory",
        "out of memory",
        "oom",
        "cudnn error: cudnn_status_alloc_failed",
        "torch.cuda.outofmemoryerror",
    ]
    if any(m in needle for m in oom_markers):
        suggested = _suggest_num_envs(current_num_envs)
        suggestions = [
            (
                f"Try `num_envs={suggested}`."
                if suggested
                else "Try a lower num_envs (e.g. halve the current value)."
            ),
            "Close other GPU-intensive processes (browser, games, other runs).",
            "Pick a smaller robot: Go1 and Cartpole fit 8 GB VRAM comfortably.",
        ]
        return CudaErrorClass(
            kind="oom",
            title="GPU out of memory",
            detail=(
                "The training run allocated more VRAM than the device has "
                "free. mjlab's parallel envs scale near-linearly with "
                "num_envs, so dropping num_envs is the usual fix."
            ),
            suggestions=suggestions,
            suggested_num_envs=suggested,
            problem_type="/problems/cuda-oom",
        )

    # 4. Driver / runtime mismatch.
    if (
        "cuda driver version is insufficient" in needle
        or "the provided ptx was compiled with an unsupported toolchain" in needle
    ):
        return CudaErrorClass(
            kind="driver_version",
            title="CUDA driver too old",
            detail=(
                "The installed NVIDIA driver is older than the CUDA "
                "toolkit mjlab was built against. Training cannot start "
                "until the driver is upgraded."
            ),
            suggestions=[
                "Upgrade NVIDIA drivers from "
                "https://www.nvidia.com/Download/index.aspx (Windows Game "
                "Ready / Studio) — WSL2 uses the Windows host's driver.",
                "Run `nvidia-smi` to confirm the new driver version "
                "before retrying.",
            ],
            problem_type="/problems/cuda-driver-too-old",
        )

    # 5. CUDA missing entirely.
    no_cuda_markers = [
        "no cuda-capable device",
        "cuda_error_no_device",
        "found no nvidia driver",
    ]
    if any(m in needle for m in no_cuda_markers):
        return CudaErrorClass(
            kind="no_cuda",
            title="No CUDA device detected",
            detail=(
                "The runtime couldn't find any NVIDIA GPU. Confirm the "
                "GPU is visible (nvidia-smi), CUDA_VISIBLE_DEVICES is "
                "not set to an empty string, and — on WSL — that the "
                "Windows host NVIDIA driver is installed."
            ),
            suggestions=[
                "Run `nvidia-smi` in the same shell that launches the backend.",
                "Unset CUDA_VISIBLE_DEVICES if it has been emptied.",
                "Switch the project to the gym_sb3 adapter to train on CPU.",
            ],
            problem_type="/problems/cuda-no-device",
        )

    return CudaErrorClass(
        kind="unknown",
        title="Training failed",
        detail="",
        suggestions=[],
    )


def _suggest_num_envs(current: Optional[int]) -> Optional[int]:
    """Heuristic: halve the current num_envs, snap to a power-of-two
    within [128, 4096]. Returns None when current is unknown."""
    if current is None or current <= 0:
        return None
    target = max(128, min(4096, current // 2))
    # Snap to next-lower power-of-two for cleaner recommendations.
    p = 1
    while p * 2 <= target:
        p *= 2
    return p
