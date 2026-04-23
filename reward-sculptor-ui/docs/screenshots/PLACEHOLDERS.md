# Screenshot placeholders

Zero-byte PNG files live next to this doc — they're markers for
`README.md` / `docs/*.md` image references so the docs don't render
broken links. Each placeholder has a suggested capture below. Replace
a placeholder with a real screenshot once the running UI shows the
relevant state.

| File | What to capture |
|------|-----------------|
| `dashboard_with_gpu.png` | Dashboard landing page, RTX 5070 Laptop card visible on the right, at least one project card on the left. |
| `library_browser.png` | Library tab with the full 63-robot grid, default filters applied (Quadruped / Humanoid / Arm / Gripper_Hand × mjlab-ready / Gymnasium). Search box empty. |
| `library_detail_modal.png` | Click into Unitree G1 — modal shows thumbnail, pre-configured tasks list (4 tasks), references list (3 papers + 2 repos with external-link icons), Create CTA. |
| `create_project_dialog.png` | CreateProjectDialog after selecting Unitree G1, mjlab adapter picked, num_envs slider at 1024, VRAM estimate green "2.0 GiB of 7.4 GiB free — comfortable headroom". |
| `coming_soon_confirm.png` | Same dialog but adapter dropdown set to `⏳ Isaac Lab (coming soon)`. Amber confirmation card with adoption-guide link visible. Submit button reads "Create anyway". |
| `mjlab_project_detail.png` | Project detail page after mjlab creation, Train button enabled, adapter_class shows MjlabAdapter, task_id visible in the adapter config view. |
| `mjlab_oom_retry.png` | CreateProjectDialog after a 412 insufficient-VRAM response at num_envs=8192. Rose OOM banner visible with "Retry with num_envs=2048" button. |
| `settings_gpu_panel.png` | Settings page scrolled to the GPU panel, live utilization + temp bars, all three adapter dots green (mjlab / mujoco_warp / rsl_rl). |
| `migration_warning_banner.png` | ProjectCard displaying the amber "Uses a deprecated adapter" banner for a hand-crafted legacy project. |

Capture checklist:

1. Start the UI: `cd ~/projects/reward-sculptor-ui && ./run.sh`.
2. Open http://localhost:5173 in a Chromium-family browser at 1440×900.
3. Use a consistent theme (light) so screenshots look uniform.
4. For GPU-related captures, have `nvidia-smi` running in a background
   terminal so VRAM figures are real.
5. Save 1440×1024 PNGs (or the actual viewport — don't manually
   resize). Commit with the same filenames as above; the zero-byte
   placeholders get overwritten.
