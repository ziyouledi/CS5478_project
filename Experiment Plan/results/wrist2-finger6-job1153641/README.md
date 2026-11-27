# wrist2-finger-6 on M1 (job 1153641, 2026-11-17)

`25_train_and_eval.sbatch`, `STUDENT_STAGE=m1`, Stage 1 PCA `1024096`, 5000 iterations, seeds 0–2; dry run `1153640` PASS. All 6 tasks COMPLETED, `train_and_eval_status=PASS` (training 1:42–1:50, evaluation 11–13 min, 25 checkpoints × 1024 episodes, seed 20261028). Recovery = (P − 0.006) / (26.11 − 0.006), final checkpoint. Welch t, uncorrected, n = 3.

| Rate | Recovery per seed | mean ± sd | falls | hand-8 (same rate) | p |
|---|---|---|---|---|---|
| 60 Hz (N = 1) | 0.610 / 0.506 / 0.842 | 0.653 ± 0.172 | 3.5% | 0.866 ± 0.112 | 0.16 |
| 10 Hz (N = 6, trained under hold) | 0.321 / 0.357 / 0.129 | 0.269 ± 0.122 | 3.3% | 0.178 ± 0.105 | 0.38 |

Same 8 numbers, wrist split vs whole-hand PCA: lower at 60 Hz, slightly higher at 10 Hz; neither difference significant. Both collapse from 60 to 10 Hz (wrist split keeps 41% of its 60 Hz value, hand-8 21%).
