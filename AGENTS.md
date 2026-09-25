# Codex Project Rules

## Language restriction

- 禁止使用“不是”两个字。

## Local search tool

- The `rg` (ripgrep) command is allowed for searching file names and file contents in this project.
- Prefer `rg` and `rg --files` over slower alternatives when searching the repository.

## Python environment

- The project's current Python/Conda environment is named `barrage`.
- Run project commands and install project dependencies in the `barrage` environment.
- Import Numba and llvmlite directly from that environment. Do not add `.runtime` to import paths or restore local dependency fallbacks. `requirements.txt` pins the currently tested versions: Numba 0.61.2 and llvmlite 0.44.0.
- Keep Numba compilation process-local with persistent JIT caching disabled. Use `python -B tools/smoke_environment_dependencies.py` for dependency compatibility checks; it blocks `.runtime` access and exercises compiled planning, RGB processing, window inference and multiprocessing collection. Smoke results do not establish performance or long-episode reliability.

## Naming directories under `runs`

- Name each new primary training output `runs/visual_set_vN`, where `N` is the model version, for example `runs/visual_set_v9`.
- A stage derived from the same version may have at most one short, standardized purpose suffix:
  - QDagger: `runs/visual_set_vN_qdagger`
  - Standalone evaluation: `runs/visual_set_vN_eval`
- Never encode hyperparameters, round counts, rebuild attempts, or temporary notes in a run directory name. Do not create chained names such as `rebuild_student_r2`, `wall_anneal_10round`, or `fixed_final_new`.
- For a new primary experiment, increment the version number, such as `visual_set_v9` to `visual_set_v10`, instead of appending more suffixes.
- Keep code defaults, README commands, checkpoint references, and evaluation outputs consistent with this convention.
- Never overwrite a non-empty run directory automatically. If the intended directory already exists, let the user decide whether to reuse it, remove it, or increment the version number.
- Do not rename historical directories under `runs` unless the user explicitly requests it, because doing so can break configs, reports, and checkpoint references.

Preferred examples:

```text
runs/visual_set_v8
runs/visual_set_v9
runs/visual_set_v9_qdagger
runs/visual_set_v9_eval
runs/visual_set_v10
```

## Artifact provenance and plotting

- Do not generate, record, save or output cryptographic hashes, file fingerprints or checksum digests in any project file or artifact. Use paths, filenames, parameters and direct content comparisons for provenance and verification. Remove active digest calculation and write logic before running affected scripts.
- All new or modified chart legends must have no border (`frameon=False` in Matplotlib, or the equivalent elsewhere), unless the user explicitly requests a border. Preserve the current `results.png` design and axis rules below.

## Local GitHub application

- The GitHub CLI executable is `D:\GitHub CLI\gh.exe`; invoke this full path when `gh` is absent from `PATH`.
- The local GitHub Copilot application executable is `D:\GitHub Copilot\github.exe`.
- This executable is the GitHub Copilot desktop application, not the GitHub CLI command `gh.exe`; do not use it as a substitute for `gh` in terminal commands.

## DAgger and QDagger evaluation and selection

- Choose evaluation scope and episode count for the current task or the user's explicit request. Do not automatically launch a fixed-200 acceptance run for latency optimization or window testing.
- Select `best.pt` for DAgger and QDagger solely by `success_at_limit`. On any exact tie, select the latest checkpoint, including ties below 1.0. `model_iqm`, `model_mean`, and confidence bounds are report-only and must not be selection keys.
- In the four-panel DAgger and QDagger `results.png` dashboards, fix the upper-right reliability panel's primary y-axis maximum at `100`. Scale its lower bound and every other plotted y-axis so the finite curve range occupies the middle 80% of the axis, leaving 10% of the height below and 10% above. Apply the same rule independently to secondary y-axes.

## Preserve the current `results.png` design

- Treat the project's current `results.png` plotting implementation as canonical. Continue using its existing panels, plotted metrics, layout, formatting, labels, colors, and overall visual style.
- Do not independently add, remove, replace, or rearrange plot content, and do not change the image format or presentation. Change the design only when the user explicitly requests that specific change.
- Preserve all explicit plotting requirements elsewhere in this file, including the existing y-axis scaling rule for DAgger and QDagger dashboards.

## Real-world observation restrictions

- During real-world testing and deployment, the model and control policy may receive only the current image frame as input.
- Do not read, inject, or otherwise use bullet or plane position, velocity, or other simulator/game state as model or policy input during real-world testing. This restriction also applies to auxiliary decision logic around the policy.

## Targeted bullet probability

- Use `targeted_bullet_probability=0.10` for all current training, evaluation, benchmarking, replay augmentation, and command-line game configurations, regardless of the total bullet count.
- Apply 10% independently whenever each bullet is initially generated or respawned; do not force exactly 10% of a finite bullet set to be targeted.

## Current deployment task

- Use 300 bullets and teacher_reaction_seconds=0.10 for current tracked DAgger training; use 300 bullets for its initial and round evaluations, configured separately with evaluation_bullet_count/--evaluation-bullets. Use 300 bullets for standalone teacher/image-teacher/tracked-policy evaluation and command-line game defaults; explicit multi-stage diagnostics retain their configured counts. Benchmarks must use the settings of the path being measured.
- Use 384 tracked-object/model slots for the 300-bullet task: 300 live bullets plus 84 slots of occlusion and temporary-track headroom. Keep this capacity to preserve compatibility with the strongest existing 384-slot checkpoints.

## Reuse and parallelize project evaluation

- Whenever evaluation is needed, use the evaluation algorithm and metric implementation already present in this project. Do not introduce a separate evaluation method or silently change its semantics.
- Compute evaluation episodes concurrently using the project's parallel evaluation path. Preserve deterministic episode assignment, fixed held-out sets, metric semantics, and checkpoint-selection rules while parallelizing.

## Evaluation metric naming

- Use `success_at_limit` as the single reported success-rate metric for training rounds. Its threshold is the configured evaluation episode limit. DAgger and QDagger checkpoint selection uses only `success_at_limit` as specified above.
- A generic uncertainty companion such as `success_at_limit_ci95_low` is allowed when a plot or report needs confidence information; it must use the same episode limit rather than introduce another success threshold.
- Do not introduce fixed-duration duplicates such as `success_at_120`, `success_at_120_count`, `success_at_120_ci95_low`, or similarly named `success_at_<seconds>` fields.
- Keep the episode-limit configuration generic, such as `evaluation_episode_limit_seconds`; do not describe it as a separate selection or success horizon.
- Do not rewrite historical artifacts under `runs` merely to remove legacy metric columns. Apply this convention to newly generated configs, manifests, summaries, histories, reports, logs, plots, and selection logic.

## Local training performance validation

### Reference development computer

The following hardware was detected on the project's current development computer on 2026-08-18 and is the required reference target for local training optimization:

- Operating system: 64-bit Windows, build `10.0.26200`.
- CPU: Intel Core Ultra 5 230F, with 10 logical processors available to the process.
- System memory: 25,460,809,728 bytes, approximately 24 GB decimal or 23.7 GiB.
- GPU: NVIDIA GeForce RTX 5060 with 8,151 MiB of VRAM.
- NVIDIA driver: `610.62`.

Treat these values as a checked-in hardware snapshot rather than dynamically guaranteed capacity. Re-detect the machine before performance work if the hardware, driver, operating system, Python environment, or training framework has changed.

### Required optimization procedure

- Whenever a training structure or training-critical path is changed, benchmark the modified data collection, simulation, preprocessing, inference, and optimization paths on the reference computer. Measure a before/after wall-clock baseline and relevant throughput, such as environment steps, samples, episodes, or optimizer batches per second.
- Tune worker count, concurrent environments, batch size, prefetching, data transfer, and other relevant parameters on this computer. Use the fastest configuration that completes correctness checks and a sustained stress run without crashes, hangs, out-of-memory failures, dropped work, corrupt artifacts, or invalid numerical results.
- Profile expensive serial operations. When iterations or tasks are independent and semantics permit it, implement and benchmark a parallel or batched path instead of leaving the work serial. Verify that configured workers truly execute concurrently and that process startup, serialization, synchronization, and I/O overhead do not erase the speedup.
- Drive CPU, GPU, and memory utilization toward the practical limit of this computer while retaining enough memory and VRAM headroom for transient peaks. Avoid CPU oversubscription, GPU out-of-memory instability, excessive context switching, and competing worker pools.
- On Windows, make multiprocessing entry points spawn-safe, initialize worker-local resources correctly, propagate failures to the parent process, and shut down workers cleanly. Preserve deterministic seed assignment across workers when determinism is required.
- Performance tuning must preserve training semantics, observation restrictions, evaluation algorithms, fixed held-out episodes, checkpoint-selection rules, outputs, and numerical invariants. Do not accept a faster result that silently changes the experiment.
- Record the tested parameters, measured throughput, resource utilization, stability duration, and observed speedup. Keep a safe fallback configuration for machines with fewer resources or for environments where the maximum-performance settings are unstable.

For long-running asynchronous work:

- Empty `write_stdin` polls MUST use `yield_time_ms >= 180000`;
  prefer `300000` when intermediate output is not needed.
- `functions.wait` MUST use `yield_time_ms >= 180000`.
- `functions.exec` MUST set its outer `@exec yield_time_ms` at least
  30000 ms longer than the longest nested tool wait, so the outer
  code cell does not yield first.
- Do not apply the long wait to non-empty `write_stdin` calls that
  send interactive input.
- These tools return early when the process or cell completes.
  Do not wake the model merely to report that work is still running.

## Active policy architecture (2026-09-25)

- Training, standalone evaluation, real-world image inference and the window share the policy/teacher-cost architecture (model version 12). Do not reintroduce a learned collision head, collision-head loss, risk calibration, learned action masking or risk-dependent analytic gates.
- Keep image-derived shared geometry and receding pixel planning. The independent analytic geometry diagnostic defaults off, matching the approved joint ablation.
- Receding pixel planning defaults to 0.5 seconds (15 decisions at 30 Hz) for training, evaluation, benchmarks and window testing. Preserve explicitly requested historical overrides and recorded experiment parameters. Teacher horizons remain independently configured.
- Current targeted evaluation and window testing share `tools/commit_safe_ranking.py`: By user request on 2026-09-24, require nominal pixel safety over the next four physics steps whenever available; otherwise maximize the nominal-safe prefix. Uncertainty remains a soft ranking signal and must not veto route eligibility. Record `ranking_variant=commit_safe_ranking` and `ranking_safety_basis=nominal_pixel` in new controller manifests.
- Window testing is pinned to `best.pt`, matching the completed 3000-episode long test, with its matching experimental controller, nominal-pixel commit-safe ranking and 0.5-second planning horizon. Training warm start remains `diagnostics/risk_removal_20260920/policy_teacher_cost.pt`. Historical round candidates stay untouched.
- New training defaults to `runs/visual_set_v52`, with 36 collection environments, 9 workers and batch 512. Legacy risk-head runs require a fresh run and explicit warm start; do not resume their optimizer or reuse their selection scores.
- Legacy output tuple slots may contain neutral constants solely for replay/student adapters. They are not trainable heads and must never affect risk filtering or gates.
- Preserve the success_at_limit selection rules. Describe the scope and limitations of each evaluation using its actual episode count.

## Deployment component discipline (2026-09-20)

- Treat additions to the real-world inference/control framework cautiously. Prefer correcting an evidenced failure in an existing component before adding another head, filter, gate, fallback or arbitration layer.
- Before proposing a new online component, identify the failure mechanism with replay evidence, explain why the existing architecture cannot adequately address it, and state the expected benefit, interaction cost and latency cost.
- Validate candidates on paired complete episode pools, reporting both rescued failures and newly introduced failures. Passing a small ablation or rescuing known failures alone does not establish a deployment improvement.
- Preserve the simplified policy/teacher-cost architecture and image-only input boundary. Keep experiments separate from deployment defaults until task-appropriate correctness and regression checks are satisfied.

## Current entry points and runtime dependencies (2026-09-25)

- Use `tools/evaluate_targeted_dagger.py` for the current `best.pt` and its matching `planner_readiness_20260921_parallel` controller. Fresh testing defaults to 3000 episodes, 9 evaluation workers, batch 20, 9 search threads and CUDA; choose the actual count for the task. Regression mode requires an explicit seed CSV. Supplemental results do not select checkpoints.
- Fresh seeds exclude the checkpoint configuration's collection and fixed held-out seeds. Do not claim disjointness from every historical evaluation pool. Report actual seeds, episode count, limit and controller configuration.
- Generic tracked-policy, teacher and image-oracle evaluation CLIs still restrict full evaluations to 200 episodes. The prepared `tools/train_targeted_dagger.py --evaluate` path also uses 200 episodes. Document these implementation limits separately from the task-scoped evaluation policy; do not start a fixed-count run merely because an entry defaults to it.
- Evaluation batch sizes are entry-specific: generic tracked evaluation defaults to 10 and the current targeted long-test entry defaults to 20. Preserve atomic intermediate and final writes.
- The generic training entry is `python -u -B -m barrage_rl.train_tracked_policy`; `train_dagger.cmd` is absent. Its defaults do not replace the separate prepared configuration used by `tools/train_targeted_dagger.py`.
- Prepared training reads `diagnostics/dagger_ready_20260921/training_config.json` and validates preflight records, launch source contents and initial model contents before `--train`. Source changes require renewed validation; do not bypass this check or silently update old approval evidence.
- The current targeted evaluation depends on the prepared training configuration and controller source files under `diagnostics/planner_readiness_20260921`. The window uses its dedicated `barrage_rl/window_runtime.py` and `barrage_rl/window_planner.py` implementation with matching controller metadata and ranking semantics. Preserve these active dependencies during diagnostics cleanup. Historical source snapshots remain historical artifacts, not active dependency fallback locations.

## Shared game and window behavior

- `barrage_rl/runtime_core.py` owns game semantics for the window, training, evaluation and tests. Keep `Barrage.py` and `barrage_rl/env.py` as adapters; do not duplicate spawning, physics, collision or rendering rules. Rule changes require deterministic state/RNG/RGB/collision comparisons and the existing local throughput checks.
- Preserve 120 Hz physics, four physics steps per current decision (30 Hz), and the 100 FPS window render cap with `tick_busy_loop`. AI and manual modes share the same game loop and array bullet storage.
- The 300-bullet opening uses ten batches of 30 at 0.1-second intervals from 0.0 to 0.9 seconds. Later batches use the plane state at their spawn time within the simulator; this state must not enter the image policy. Respawns preserve per-bullet independent targeting draws.
- Window inference stays on CPU, loads version-12 `best.pt`, and installs the matching experimental controller through `barrage_rl/window_runtime.py`. New training rounds must not automatically replace the window checkpoint.
- Window RGB copying and foreground scanning default to four Numba threads. `--rgb-workers 0` or `1` uses the serial fallback; restore the planner thread mask after processing. Training and calls outside the window RGB context retain serial behavior.
- Preserve the dedicated `WindowImageTracker` association precision, tie order, velocity fitting, occlusion behavior and NumPy fallback for unsupported inputs.
- Use `tools/test_visible_window.py` for visible-window latency checks. `--no-ai` selects manual mode; `--immune` explicitly disables collision damage for timing. Do not interpret latency smoke output as a complete episode evaluation.
