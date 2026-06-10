# Remote GPU dispatch (Ship 23)

Dispatch the GPU-heavy mjlab **train** step (and, opt-in, **rollout**)
to a rented pod over SSH+rsync. Everything else — diagnose, edit, KG,
criteria, the UI — runs locally, untouched. On a RunPod Community 5090
(~$0.69/hr) a train iteration that takes ~25 min on the 8 GiB RTX 5070
laptop should land around ~5 min.

## How it works (one paragraph)

mjlab training was already a self-contained subprocess
(`python -m sculptor.adapters._mjlab_runner train ...`) whose inputs
are small files and whose outputs are files in `output_dir`. With
`[remote]` enabled, `MjlabAdapter.train()` instead uploads the inputs
to absolute-path-mirrored locations under `~/.sculptor_remote/mirror/`
on the pod, rsyncs the live `sculptor/` source to `~/.sculptor_remote/
code/` (PYTHONPATH points there — local/remote sculptor skew is
impossible), launches the runner detached (`setsid`, file-backed
`pgid`/`stdout.log`/`stderr.log`/`exitcode`), polls one combined ssh
round-trip per 5 s (exit code + new stdout, re-emitted locally so the
UI sees live `iter_progress`), then downloads artifacts into a staging
dir and promotes them with `checkpoint.pt` (the resume key) moved LAST.
All existing post-checks, resume logic, and error formatting run
unchanged. See `sculptor/adapters/_remote.py` for the full protocol.

## RunPod setup (one-time, ~10 min)

1. **Account**: runpod.io → add a payment method. Costs are per-second.
2. **(Recommended) Network volume**: Storage → New Network Volume,
   50 GB (~$3.50/mo), in a datacenter that lists RTX 5090s. Mount path
   `/workspace`. This keeps the provisioned venv + mirrored checkpoints
   across pod restarts — without it you re-provision (~5 min) per pod.
3. **Pod**: Deploy → Community Cloud → GPU = RTX 5090 (do NOT pick
   A100/H100 — mujoco_warp is plain-FP32-bound and a 5090's 105 FP32
   TFLOPS beats an A100's 19.5). Template: any recent
   `runpod/pytorch:*cuda12.8*` image (driver must be ≥ R570 — the
   provisioning script verifies). Attach the network volume. Expose
   SSH (default on RunPod: it prints an `ssh root@<ip> -p <port>`
   line + injects your account's public key).
4. **SSH key**: RunPod Settings → SSH keys → paste `~/.ssh/id_ed25519.pub`
   (generate inside WSL with `ssh-keygen -t ed25519` if needed).

If you use the network volume, provision with
`-w /workspace/sculptor_remote` (below) so the mirror + wheel cache live
on it, and set the matching `remote_workdir` in the config — the script
prints the right block either way.

The **venv always goes on pod-local disk** (`~/.sculptor_venv`), never
the volume: RunPod's network fs does not page-cache, and a
volume-resident venv costs ~60 s of import I/O (torch alone 26–39 s) in
every runner subprocess — measured live, it turned a 29 s train into a
199 s job. After a pod restart the venv is gone: re-run the provision
script (fast — wheels come from the volume cache) and update host/port
in the UI's Settings → Remote GPU card.

## Provision the pod

From WSL, inside `RewardSculptor/` (so local torch/mjlab versions get
pinned remotely — this keeps the version-skew check green):

```bash
./scripts/provision_remote.sh root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519 \
    [-w /workspace/sculptor_remote]   # only with a network volume
```

Idempotent — re-run freely. It installs rsync/uv/python-3.13 venv +
pinned `torch` / `mjlab[cu128]` / `imageio-ffmpeg`, sanity-checks the
GPU + driver, and prints the `[remote]` block to paste into your
project's `config.toml`.

## Configure

Top level of the project's `config.toml` — **not** inside
`[adapter].config`:

```toml
[remote]
enabled = true
host = "203.0.113.7"
port = 41234
user = "root"
key_path = "~/.ssh/id_ed25519"
remote_python = "~/.sculptor_venv/bin/python"
# remote_workdir = "~/.sculptor_remote"   # set to /workspace/... on a network volume
# device = "cuda:0"          # device ON THE POD (defaults to adapter device)
# rollout_remote = false     # rollouts stay local by default (video preview robustness)
# poll_interval_s = 5.0
```

Every connection field has a `SCULPTOR_REMOTE_*` env override that WINS
over the TOML (`ENABLED`, `HOST`, `PORT`, `USER`, `KEY_PATH`, `WORKDIR`,
`PYTHON`, `DEVICE`, `ROLLOUT`) — that's also how the UI backend injects
settings (Ship 23d). The tuning knobs (`connect_timeout_s`,
`poll_interval_s`, `reattach_max_failures`) are TOML-only. With remote enabled the adapter skips the
local-VRAM `num_envs` autocap, so `num_envs = 4096` is honored even
when configured from the 8 GiB laptop.

Verify before running anything:

```bash
uv run sculpt remote doctor --config examples/<proj>/config.toml
```

checks: local ssh/rsync, host reachable, remote rsync/python, NVIDIA
driver ≥ R570 + GPU, `torch.cuda.is_available()`, torch/mjlab version
skew vs local, free disk. `--json` for machine-readable output. Exit
codes: 0 all green, 1 something failed, 2 nothing configured.

## Failure modes (all observable as `[SCULPT-EVENT]` lines / UI events)

| event | meaning / behavior |
|---|---|
| `remote_dispatch_started`, `remote_upload_completed`, `remote_job_launched`, `remote_job_finished`, `remote_artifacts_synced` | normal lifecycle |
| `remote_job_reattached` | a still-running remote job matched this dispatch's command hash (local crash + `--resume`) — reattached instead of double-training; upload skipped |
| `remote_stale_job_killed` | a running job with a DIFFERENT command was killed before launch |
| `remote_connection_lost` | ssh poll failed; backs off and retries (~60 attempts) — the detached job survives |
| `remote_version_skew` | remote torch/mjlab ≠ local; warns and proceeds (fix via provision script) |
| `remote_config_ignored` | `[remote]` present but unusable (typo'd keys / enabled without host) — training runs LOCALLY, once-per-process warning |
| `remote_dispatch_failed` reason=`ssh_unreachable` | host gone; job left running for reattach |
| … reason=`launch_failed` | could not write/launch run.sh or read its pgid; retry reattaches if the job actually started |
| … reason=`remote_oom` | classified from remote stderr; flows into the normal recoverable iteration-failure path |
| … reason=`runner_failed` | nonzero runner exit; remote stderr tail attached |
| … reason=`artifacts_missing` / `sync_failed` | download problems; staging dir discarded — a partial sync can never look complete |
| … reason=`job_lost` | the job dir vanished (spot reclaim / pod wipe); fails fast, stage retrains |

Kill semantics mirror local training: any local exception (UI cancel,
Ctrl-C) SIGTERMs the remote process group — except when the host is
unreachable, in which case the job is deliberately left alive for
reattach. The on-pod `exitcode` file, never the SSH channel, decides
whether a job finished.

## Measured (Ship 23e, RunPod Community RTX 5090, 2026-06-09/10)

G1 humanoid (`Mjlab-Velocity-Flat-Unitree-G1`), intrinsic reward, 50
rsl_rl iters, seed-pinned:

| | envs | s/iter | steps/s | 50-iter job wall |
|---|---|---|---|---|
| local RTX 5070 Laptop (8 GiB, autocap) | 2048 | 1.08–1.15 | ~45k | 71.5 s |
| remote RTX 5090 (32 GiB) | 4096 | 0.65 | ~150k | 117 s (62 s train + ~48 s startup + ~7 s sync) |

**3.3× sample throughput.** Per-dispatch overhead ≈ 48 s startup
(imports + 4096-env build) + ~13 s upload/sync; amortized over a real
1500-iter stage: remote ≈ 17 min vs local ≈ 28 min at 2× the
batch — ~3× experience throughput, ~1.6× wall at same iter count.
Cartpole sanity (256 envs): 57k steps/s, 10 iters in 12.6 s.
Verified live: warm-start from the mirror (zero-byte re-upload),
`kill -9` of the local process → `remote_job_reattached`, no
double-train; SIGTERM (UI cancel) → pod GPU freed in <5 s; doctor all
8 checks green through `POST /system/remote/doctor`.

## Manual smoke checklist (Ship 23e — run on a real pod)

1. `./scripts/provision_remote.sh root@<ip> -p <port>` → prints config block.
2. `uv run sculpt remote doctor --config <cfg>` → all green, exit 0.
3. 1-iter run: `uv run sculpt run "<goal>" --config <cfg> --iterations 1`
   with `[remote] enabled = true` → watch for `remote_job_launched` →
   `iter_progress` lines streaming → `remote_artifacts_synced`;
   `checkpoint.pt` + `metrics.json` + `reward_trajectory.json` land in
   the iter dir; rollout runs locally on the synced checkpoint.
4. Kill-mid-train: start another run, Ctrl-C during training; on the
   pod `nvidia-smi` shows the job died (kill-on-exception). Re-run with
   `--resume`: a NEW train starts (no stale state confusion).
5. Reattach: start a run, `kill -9` the local sculpt process (no
   cleanup), re-run with `--resume` → `remote_job_reattached`, no
   double-train, artifacts sync at the end.
6. UI: start a mission from the UI with remote enabled → timeline shows
   the remote_* events; cancel from the UI → pod job dies.
7. Record wall-clock per iteration + $ in CONTEXT.md (target ≤ ~1/4 of
   local).
8. **Stop the pod** (or rely on per-second billing + auto-stop).
