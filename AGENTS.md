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
