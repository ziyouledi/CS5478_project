"""Does offline staleness (step 0, job 1110496) predict trained Recovery (39-run grid)?

CPU only, existing summaries. One point per (intent, hold N): x = offline hold
metric of the signal on the expert rollout, y = Recovery retained relative to the
same intent at N = 1 (seed mean). Compares predictors by Spearman rho and by the
leave-one-intent-out error of a one-parameter fit retained = exp(-a x).
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--hold_csv", default="results/intent-hold-job1110496/intent_hold.csv")
p.add_argument("--runs_csv", default="results/rate-sweep-grid-39runs/runs.csv")
p.add_argument("--out", default="results/rate-sweep-staleness")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

hold = pd.read_csv(a.hold_csv).rename(columns={"hold_steps": "hold"})
runs = pd.read_csv(a.runs_csv)
g = runs.groupby(["intent", "hold"]).agg(recovery=("recovery", "mean"), sd=("recovery", "std"),
                                         n=("recovery", "size"), ceiling=("ceiling", "first")).reset_index()
assert len(g) == 16 and (g.n == 3).all(), g
base = g[g.hold == 1].set_index("intent").recovery
g["retained"] = g.recovery / g.intent.map(base)
g["retained_vs_ceiling"] = g.recovery / (g.intent.map(base) * g.ceiling)
g = g.merge(hold, on=["intent", "hold"], how="left", validate="1:1")
assert g.normalized_hold_error.notna().all()
g["log2_hold"] = np.log2(g.hold)
g["stale_tol"] = g.fraction_angle_error_above_success_tolerance  # quaternion intents only
d = g[g.hold > 1].copy()  # N = 1 is retained = 1 at x = 0 by construction

def loio(x, y, groups):
    """Leave-one-intent-out RMSE of retained = exp(-a x)."""
    err = []
    for k in np.unique(groups):
        tr, te = groups != k, groups == k
        f = lambda s: np.sum((np.exp(-s * x[tr]) - y[tr]) ** 2)
        s = optimize.minimize_scalar(f, bounds=(0, 100), method="bounded").x
        err.extend(np.exp(-s * x[te]) - y[te])
    return float(np.sqrt(np.mean(np.square(err))))

out = {"n_points": int(len(d)), "predictors": {}}
for y in ["retained", "retained_vs_ceiling"]:
    for x in ["normalized_hold_error", "log2_hold"]:
        rho, pv = stats.spearmanr(d[x], d[y])
        out["predictors"][f"{y}~{x}"] = {"spearman": float(rho), "p": float(pv),
            "loio_rmse": loio(d[x].values, d[y].values, d.intent.values)}
q = d[d.stale_tol.notna()]
rho, pv = stats.spearmanr(q.stale_tol, q.retained)
out["predictors"]["retained~stale_tol (obj-4, goal-4 only)"] = {"spearman": float(rho), "p": float(pv), "n": int(len(q))}
out["table"] = g[["intent", "hold", "recovery", "sd", "ceiling", "retained", "retained_vs_ceiling",
                  "normalized_hold_error", "stale_tol", "action_variance_explained"]].round(4).to_dict("records")
json.dump(out, open(f"{a.out}/summary.json", "w"), indent=1)
g.to_csv(f"{a.out}/points.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
col = {"hand-2": "#c44", "hand-8": "#e80", "obj-4": "#284", "goal-4": "#26c"}
for it, s in g.groupby("intent"):
    s = s.sort_values("hold")
    for k, (x, xl) in enumerate([("normalized_hold_error", "offline hold error / variance (step 0)"),
                                 ("stale_tol", "share of steps outside 0.1 rad (step 0)")]):
        if s[x].notna().all():
            ax[k].plot(s[x], s.retained, "o-", c=col[it], label=it)
            for _, r in s.iterrows():
                ax[k].annotate(f"N={int(r.hold)}", (r[x], r.retained), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[k].set_xlabel(xl)
    ax[2].plot(60 / s.hold, s.retained, "o-", c=col[it], label=it)
ax[2].set_xscale("log"); ax[2].set_xlabel("update rate (Hz)")
for x in ax:
    x.set_ylabel("Recovery retained vs same intent at 60 Hz"); x.grid(alpha=.3); x.legend()
ax[0].set_title("staleness as predictor"); ax[1].set_title("quaternion intents: tolerance metric"); ax[2].set_title("rate as predictor")
fig.tight_layout(); fig.savefig(f"{a.out}/retained_vs_staleness.png", dpi=150)
print(g[["intent", "hold", "recovery", "retained", "retained_vs_ceiling", "normalized_hold_error", "stale_tol"]].round(3).to_string(index=False))
print(json.dumps(out["predictors"], indent=1))
print("staleness_scatter_status=PASS")
