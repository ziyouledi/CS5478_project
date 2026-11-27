# M1 hand-8 at 10 Hz (intent hold 6) resumed 5000 → 10000 — job 1162543_0–4

Source: `1162543_0–4` (`25_train_and_eval.sbatch` via `cluster_smoke/submit_m1_hand8_hold6_resume.sh`), all COMPLETED 0:0, `train_and_eval_status=PASS`, 1:55–2:00 each. Resumed from the newest `ep_5000` checkpoint of `1111584_6–8` (seeds 0–2) and `1123488_8–9` (seeds 3–4). Evaluation: Stage 1 task, goal hidden, intent hold 6 (as trained), 1024 episodes per checkpoint, checkpoints 5200–10000 (25 each). Restore check: 5200 values continue the 5000 values (within 0.2 successes).

Successes per episode; recovery = (P − 0.006) / (26.11 − 0.006).

| seed | 5000 | 6000 | 8000 | 9000 | 10000 | mean 9200–10000 | rec 5000 | rec 10000 | rec mean 9200–10000 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.69 | 7.75 | 7.25 | 6.89 | 6.67 | 6.22 | 0.295 | 0.255 | 0.238 |
| 1 | 2.36 | 2.90 | 3.54 | 3.80 | 4.03 | 3.87 | 0.090 | 0.154 | 0.148 |
| 2 | 3.91 | 4.40 | 5.76 | 5.84 | 5.21 | 5.49 | 0.149 | 0.200 | 0.210 |
| 3 | 7.77 | 7.98 | 8.78 | 8.91 | 8.73 | 8.96 | 0.298 | 0.334 | 0.343 |
| 4 | 2.66 | 2.95 | 3.39 | 3.50 | 3.65 | 3.54 | 0.102 | 0.140 | 0.135 |
| mean | | | | | | | 0.187 | 0.217 | 0.215 |

hand-8 at 60 Hz (5000 iterations, `../phase2-block4-5seeds/runs.csv`): 0.879 / 0.748 / 0.970 / 0.973 / 0.414, mean 0.797.

Reading: doubling training moves the 10 Hz mean from 0.19 to 0.22 and all five seeds flatten between 8000 and 10000 (change over the last 1000 iterations −0.62 to +0.23 successes). The best 10 Hz seed after 10000 iterations (0.34) is still below the worst 60 Hz seed after 5000 (0.41): complete separation (U = 0), even with twice the training budget at 10 Hz. The M1 10 Hz collapse is not a training-length artifact.
