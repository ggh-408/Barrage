# Codex Project Rules

Apply the global task settings in [README.md](README.md) across training, evaluation, benchmarks, replay augmentation and the window. Entry points, warm-start sources and experimental dependencies are documented in [technical notes](docs/technical.md). The rules below govern development; later sections supplement rather than repeat the global rules.

## Global rules

- 禁止使用“不是”两个字。
- Do not generate, output or save hashes, file fingerprints or checksum digests. Use paths, parameters and direct content comparisons; remove digest calculation and writing before running affected scripts.
- Real-world policies and auxiliary decision logic may receive only the current image frame. Never feed simulator positions, velocities or other game state into deployment decisions.
- Run project commands and dependency installation in the `barrage` environment. Import dependencies directly from that environment; versions are pinned in `requirements.txt`. Disable persistent Numba JIT caching.
- Prefer `rg` and `rg --files` for searches. If GitHub CLI is absent from PATH, use `D:\GitHub CLI\gh.exe`; `D:\GitHub Copilot\github.exe` is a separate desktop application.

## Model and game invariants

- Preserve the version-12 policy/teacher-cost architecture, shared image geometry and receding pixel planning. Reject other or missing checkpoint versions and unexpected parameters; do not restore learned collision heads, risk losses, calibration, action masking or risk-dependent gates.
- Keep the model capacity and planning settings in README consistent across all execution paths. Explicit experiment overrides remain recorded separately; teacher horizons are independently configured. Neutral replay/student output slots must never influence learning, filtering or gates.
- Implement the documented nominal-pixel ranking through `tools/commit_safe_ranking.py`. Uncertainty is a soft ranking signal and must not veto route eligibility. Record `ranking_variant=commit_safe_ranking` and `ranking_safety_basis=nominal_pixel` in controller manifests.
- Keep tracked DAgger's teacher_reaction_seconds=0.10. Configure evaluation bullet count independently; manifests and round evaluations must match their experiment. Preserve explicit diagnostic configurations, and benchmark the measured path's actual settings. Size and speed generalization are outside the training target.
- `barrage_rl/runtime_core.py` exclusively owns spawning, physics, collision and rendering. Keep `Barrage.py` and `barrage_rl/env.py` as adapters. Rule changes require deterministic state/RNG/RGB/collision comparisons and local throughput checks.
- Preserve 120 Hz physics, four physics steps per decision and the README window render cap. AI and manual modes share the game loop and array bullet storage. Opening generation uses ten equal batches at 0.1-second intervals; later batches use the simulator plane state at their spawn time.
- Preserve `WindowImageTracker` precision, tie order, velocity fitting, occlusion and NumPy fallback. Window RGB processing defaults to four Numba threads; settings 0 or 1 use the serial fallback. Restore the planner thread mask afterward; training and calls outside the window context remain serial.
- Keep the pinned window checkpoint and matching controller; new training rounds must not replace them automatically. Preserve experiment candidates and active diagnostic dependencies. Historical source snapshots must not become runtime fallbacks.
- Prepared training (`tools/train_targeted_dagger.py`) starts from the root `best.pt` with a fresh optimizer. Revalidate the current source and checkpoint before refreshing approval records and launch snapshots; remove temporary validation artifacts after the checks. The generic trainer keeps its separately documented warm-start default.
- Preserve prepared-training preflight checks. Source changes require renewed validation; never bypass checks or silently update old approval evidence.

## Evaluation and reporting

- Choose episode counts and scope for the task. Entry-specific limits do not authorize extra evaluation. Reuse existing evaluation algorithms and metrics through the parallel path, preserving deterministic episode assignment and fixed held-out pools.
- For DAgger and QDagger, select `best.pt` solely by `success_at_limit`, using the configured episode limit. Every exact tie selects the latest checkpoint, including ties below full success. Other metrics and confidence bounds are report-only; supplemental tests do not select checkpoints.
- Use generic fields such as `evaluation_episode_limit_seconds` and `success_at_limit_ci95_low`; do not add duration-specific success metrics or a separate selection horizon. Apply this consistently to new configurations, reports, logs, plots and selection logic.
- Fresh seeds exclude the checkpoint's collection and fixed held-out seeds; do not claim exclusion of all historical evaluation pools. Regression tests require an explicit seed CSV. Report actual seeds, task parameters, episode count, limit and controller configuration; keep intermediate and final writes atomic.
- Keep completed long-test evidence separate from optimized-runtime checks. Use README's measured results and limitations; smoke checks establish compatibility only, not performance or long-episode reliability.
- Window timing uses `tools/test_visible_window.py` with damage enabled by default. Enable immunity only when explicitly requested, and distinguish post-death timing from live gameplay.

## Plots

- All new or modified legends must be borderless (`frameon=False` or equivalent), unless explicitly requested otherwise.
- Preserve the existing `results.png` panels, metrics, layout, labels, colors and image format. Change the design only when explicitly requested.
- In DAgger and QDagger dashboards, fix the upper-right reliability panel's primary y-axis maximum at 100. Choose its lower bound and all other axis ranges so finite curves occupy the middle 80%, with 10% padding below and above. Apply this independently to secondary axes.

## Performance and deployment changes

- Reference hardware: Windows build 10.0.26200, Core Ultra 5 230F with 10 logical processors, about 24 GB RAM, RTX 5060 with 8,151 MiB VRAM, driver 610.62. Re-detect before performance work if the machine, driver, environment or framework changed.
- Changes to training structure or critical paths require before/after wall-clock and throughput benchmarks for affected collection, simulation, preprocessing, inference and optimization. Profile serial bottlenecks and benchmark parallel or batched alternatives where semantics permit; verify real concurrency and account for startup, transfer and synchronization overhead.
- Tune workers, environments, batches, prefetch and transfers on the reference machine. Use the fastest configuration that passes correctness and sustained stress checks without crashes, hangs, memory exhaustion, dropped work, corrupt artifacts or invalid numbers. Preserve memory headroom and avoid oversubscription; retain a fallback for smaller machines.
- Windows multiprocessing must be spawn-safe, initialize worker-local resources, propagate errors, preserve deterministic seeds and shut down cleanly. Record parameters, throughput, resource use, stress duration and speedup without changing experiment semantics or outputs.
- Prefer fixing evidenced failures in existing components. Before adding an online head, filter, gate or fallback, provide replay evidence, explain why existing components are insufficient and assess benefit, interaction and latency costs. Validate on paired complete episode pools, reporting rescued and newly introduced failures; keep candidates separate from deployment until regression checks pass.
- Use `tools/smoke_environment_dependencies.py` for dependency checks covering compiled planning, RGB processing, window inference and multiprocessing collection.

## Long-running work

- Empty `write_stdin` polls and `functions.wait` use `yield_time_ms >= 180000`; prefer 300000 when intermediate output is unnecessary. This does not apply to interactive input.
- Set the outer `functions.exec` yield at least 30000 ms longer than the longest nested wait. Tools return early on completion; do not poll solely to repeat that work is running.
