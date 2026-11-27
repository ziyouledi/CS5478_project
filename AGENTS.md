# AGENTS.md

## Scope and context

This Mac repository is the source of truth for source-code changes. NUS SoC Slurm runs compute-heavy experiments. For Shadow Hand experiment work, read [CONTEXT.md](CONTEXT.md) before planning or executing. Use [HANDOFF.md](HANDOFF.md) for the historical results and data layout, and [the v3 experiment plan](Docs/Experiment%20Plan/低维意图接口实验设计%20v3（分阶段）.md) for scientific design. Current user instructions and verified new results supersede stale dates or status in those documents.

## Cluster operations

- Use the `soc-cluster` SSH alias only for lightweight login-node work: inspect files and jobs, sync code/configuration, submit Slurm jobs, and read logs or small results. Submit simulation, training, GPU work, and large analyses through Slurm. Do not SSH directly into compute nodes.
- Use only non-interactive SSH from the agent:

  ```bash
  ssh -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no soc-cluster '<remote command>'
  ```

  The first cluster operation in a task may check `hostname` and `squeue -u "$USER"` this way. A header-only queue means no active jobs. If the check fails, report the failure and continue with local work; do not request credentials or alter authentication settings. Never read, print, store, or attempt to obtain credentials or private keys.
- Keep login-node work lightweight. Inspect active jobs with `squeue -u "$USER"`; inspect finished jobs with a job-scoped `sacct -j <JOB_ID>` and their stdout/stderr. Submit with `sbatch` only when the task or experiment plan authorizes that run. Cancel only a specific job belonging to the current work when justified; broad cancellation requires explicit user instruction.
- Before syncing source files over the cluster copy, inspect both sides if remote edits may exist. Preserve completed results and checkpoints. Keep experiment outputs outside Git, in the cluster data directories described in CONTEXT.md.

## Experiment execution

1. Identify the condition, seed, checkpoint, configuration, resource request, acceptance criteria, and output path before submitting a job. Keep the planned scientific comparison fixed across conditions.
2. Review the local diff and run appropriate lightweight checks. Sync only required files. Submit the Slurm script; record its job ID and condition, and confirm submission succeeded.
3. Treat jobs as asynchronous. Do not occupy a terminal with `tail -f` or a long sleep while training runs. On a later invocation, inspect `sacct`, stdout/stderr, metrics, and output manifests before marking a result complete.
4. Diagnose failures by layer: SSH/network, Slurm/resources, code/runtime, experiment, or evaluation. Inspect evidence before resubmitting. `COMPLETED`, `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `CANCELLED` require different responses. Record failed attempts as well as successful ones.

The v3 plan is the scientific specification. Keep reward, physics, observation/action meaning, evaluation metrics, success criteria, dataset definition, checkpoint compatibility, and resource limits fixed unless the user authorizes a change. Fix clear implementation or infrastructure errors within the assigned task; ask before changing scientific semantics or adding hypotheses.

## Code and reporting

- Inspect relevant code and conventions before editing. Keep changes scoped; avoid unrelated refactors. After editing, inspect the diff and verify behavior in proportion to the risk. Distinguish existing failures from new ones.
- Store code, configuration, small result summaries, and provenance in Git. Leave `.npy` datasets, `.pth` checkpoints, full logs, runs, and caches on the cluster. Check staged files before committing.
- Update [CONTEXT.md](CONTEXT.md) when new evidence changes the active status, next step, selected checkpoint, or deadline. Add detailed historical results to [HANDOFF.md](HANDOFF.md) or the experiment record. Cite the job ID and artifact path for any completion claim. A submitted job or written script alone is not an experimental result.

## Save Context to Memory
Sessions stay one task long. When a task finishes and context is past ~100k, or the next request is unrelated to it, end the reply by suggesting `/clear`. Mid-task past ~200k, save the state and suggest `/clear`. Keep multi-step plans in a file so each step can start in a fresh session. Before suggesting `/clear`, first save the state yourself in the same turn (project memory's state note for done/next/decisions; a file for a plan too big for it), then say it is saved; never make the user ask. Suggest `/clear` only at those moments, once, not after every reply, so state is written only when it is needed.
