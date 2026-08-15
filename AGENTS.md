# Codex Project Rules

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

## Local GitHub application

- The local GitHub Copilot application executable is `D:\GitHub Copilot\github.exe`.
- This executable is the GitHub Copilot desktop application, not the GitHub CLI command `gh.exe`; do not use it as a substitute for `gh` in terminal commands.

## Evaluation metric naming

- Use `success_at_limit` as the single success-rate metric for training rounds and checkpoint selection. Its threshold is the configured evaluation episode limit.
- A generic uncertainty companion such as `success_at_limit_ci95_low` is allowed when a plot or report needs confidence information; it must use the same episode limit rather than introduce another success threshold.
- Do not introduce fixed-duration duplicates such as `success_at_120`, `success_at_120_count`, `success_at_120_ci95_low`, or similarly named `success_at_<seconds>` fields.
- Keep the episode-limit configuration generic, such as `evaluation_episode_limit_seconds`; do not describe it as a separate selection or success horizon.
- Do not rewrite historical artifacts under `runs` merely to remove legacy metric columns. Apply this convention to newly generated configs, manifests, summaries, histories, reports, logs, plots, and selection logic.
