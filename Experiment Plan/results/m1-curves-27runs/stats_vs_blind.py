import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
out = Path(sys.argv[1])
P_E = 26.1103515625
rows = list(csv.DictReader((out / "runs_successes.csv").open()))
by = {}
for r in rows:
    by.setdefault(r["intent"], []).append(r)
blind_final = np.array([float(r["final_normalized"]) * P_E for r in by["blind"]])
P_B = blind_final.mean()
res = {"P_E": P_E, "source": "27 runs: train 1104420, eval 1104422 + 1104730 + 1108150", "P_blind_mean": P_B, "test": "Welch t vs blind, uncorrected, 3 seeds per condition",
       "recovery_definition": "(final successes - mean blind final) / (P_E - mean blind final), per seed", "conditions": {}}
for intent, rs in by.items():
    final = np.array([float(r["final_normalized"]) * P_E for r in rs])
    auc = np.array([float(r["auc_normalized"]) for r in rs])
    rec = (final - P_B) / (P_E - P_B)
    c = {"seeds": [int(r["seed"]) for r in rs], "final": final.tolist(),
         "recovery_mean": rec.mean(), "recovery_sd": rec.std(ddof=1),
         "auc_mean": auc.mean(), "auc_sd": auc.std(ddof=1),
         "n80": [r["n80_iterations"] or None for r in rs]}
    if intent != "blind":
        blind_auc = np.array([float(r["auc_normalized"]) for r in by["blind"]])
        c["p_final_vs_blind"] = float(stats.ttest_ind(final, blind_final, equal_var=False).pvalue)
        c["p_auc_vs_blind"] = float(stats.ttest_ind(auc, blind_auc, equal_var=False).pvalue)
    res["conditions"][intent] = c
(out / "stats_vs_blind.json").write_text(json.dumps(res, indent=2) + "\n")
print(f"{'intent':16} {'recovery':>15} {'AUC':>13} {'p_final':>8} {'p_auc':>8}  N80")
for k, c in res["conditions"].items():
    print(f"{k:16} {c['recovery_mean']:.3f}±{c['recovery_sd']:.3f}  {c['auc_mean']:.3f}±{c['auc_sd']:.3f} "
          f"{c.get('p_final_vs_blind', float('nan')):8.3g} {c.get('p_auc_vs_blind', float('nan')):8.3g}  {c['n80']}")
