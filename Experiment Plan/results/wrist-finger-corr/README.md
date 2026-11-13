# Wrist–finger correlation of expert actions (2026-11-13)

CPU only, no Slurm job. Input: action-PCA bases `1024096` (Stage 1/M1, rollout `1023537`) and `1033179` (M2, rollout `1033178`); SHA-256 in `input_sha256.txt` matches the cluster copies. The full 20/20 eigendecomposition gives the exact sample covariance of the 20-D expert actions (1,228,800 steps each). Check: the rebuilt finger block reproduces the stored 18-D finger eigenvalues to 1e-15. Script `wrist_finger_corr.py`, full output `output.txt`.

| | Stage 1 (M1 task) | M2 (z-axis) |
|---|---|---|
| Wrist share of total action variance | 15.5 % | 11.9 % |
| Mean \|r\| wrist→finger (WRJ1 / WRJ0) | 0.11 / 0.15 | 0.20 / 0.22 |
| Mean \|r\| finger→finger | 0.14 | 0.18 |
| R² of each wrist action from 18 fingers (WRJ1 / WRJ0) | 0.33 / 0.43 | 0.68 / 0.53 |
| Canonical correlations wrist vs fingers | 0.72, 0.50 | 0.84, 0.71 |

Equal-budget variance captured (Stage 1; matches M1 recovery order):

| Numbers | hand-k | wrist split | M1 recovery hand / split |
|---|---|---|---|
| 2 | hand-2 35.8 % | wrist-2 15.5 % | 0.50 / 0.29 |
| 4 | hand-4 56.8 % | wrist2-finger-2 49.9 % | 0.71 / 0.55 |
| 8 | hand-8 81.1 % | wrist2-finger-6 78.1 % (not trained) | 0.87 / — |

Reading: unlike human grasping (Liu 2016; Jarque-Bou 2019), the expert's wrist commands are about as correlated with the fingers as the fingers are with each other. The wrist carries only 12–16 % of the variance, so spending 2 numbers on it uncompressed captures less than the same budget of whole-hand PCs. Caveats: linear, same-step correlation of saturated (bang-bang) action commands, not joint motion; variance captured is not task information; recovery differences are not significant (p = 0.28, 0.31, 3 seeds).
