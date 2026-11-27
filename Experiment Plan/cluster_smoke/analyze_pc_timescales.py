#!/usr/bin/env python3
"""Time structure of each action principal component on the recorded expert rollout (CPU, no Isaac Sim).

Signals: z_i = W[:, i].T (a - mu) for the 20 principal components of the action PCA (hand-k sends z_1..z_k),
and, for comparison, the 20 raw action dimensions. Per signal, within episodes:

* autocorrelation at each lag, the first lag where it falls below 0.5 and below 1/e (linear interpolation),
  and the integrated autocorrelation over the tested lags;
* zero-order hold error E[(z_t - z_held)^2] / var for each hold N (refresh when episode step % N == 0);
  1 = no better than sending the mean, 2 = uncorrelated with the held value;
* step change: mean |z_t - z_{t-1}| / sd, and how often the centred signal changes sign;
* goal share: fraction of the signal's variance explained by its mean over each goal segment (the steps
  between two goal changes of one environment). Null: the same with the segments of the neighbouring
  environment (same lengths, unaligned); any autocorrelated signal has a null above segments / samples.
* excess kurtosis (-2 = two-point, 0 = Gaussian).

Reference: the slowest linear directions of the action (slow feature analysis: minimise var(dz) / var(z)),
to see whether any linear combination of joints is slow, whatever its variance.
The data are the expert's own trajectories: a property of the signals, not of a student.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from analyze_intent_hold import GOAL_CHANGE_ANGLE, held_index, quat_angle, write_json
from intent_modules import GOAL_ROT, OBS_DIM, sha256

ACTION_DIM = 20
CONTROL_HZ = 60.0
HAND_KS = (2, 4, 8)


def crossing(lags: list[int], acf: np.ndarray, level: float) -> float | None:
    """First lag (interpolated, lag 0 = 1) where the autocorrelation is below `level`; None if never."""
    xs, ys = [0, *lags], [1.0, *acf]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y1 < level:
            return float(x0 + (y0 - level) / (y0 - y1) * (x1 - x0))
    return None


def signal_stats(z: np.ndarray, episode_ids: np.ndarray, episode_steps: np.ndarray, segments: np.ndarray,
                 lags: list[int], holds: list[int]) -> list[dict]:
    """z: [T, E, D]. One dict per column."""
    dims = z.shape[-1]
    flat = z.reshape(-1, dims)
    mean, var = flat.mean(axis=0), flat.var(axis=0)
    safe_var = np.where(var > 1e-12, var, np.nan)
    centred = z - mean

    acf = np.empty((len(lags), dims))
    for j, lag in enumerate(lags):
        same = episode_ids[:-lag] == episode_ids[lag:]
        acf[j] = (centred[:-lag] * centred[lag:])[same].mean(axis=0) / safe_var
    hold_error = np.empty((len(holds), dims))
    for j, hold in enumerate(holds):
        index, valid = held_index(episode_steps, episode_ids, hold)
        held = np.take_along_axis(z, index[..., None], axis=0)
        hold_error[j] = ((z - held) ** 2)[valid].mean(axis=0) / safe_var

    same = episode_ids[:-1] == episode_ids[1:]
    step = np.abs(z[1:] - z[:-1])[same].mean(axis=0) / np.sqrt(safe_var)
    sign_switch = ((centred[1:] * centred[:-1]) < 0)[same].mean(axis=0)

    def segment_share(labels: np.ndarray) -> np.ndarray:
        seg = labels.reshape(-1)
        counts = np.bincount(seg).astype(np.float64)
        share = np.empty(dims)
        for d in range(dims):
            seg_mean = np.bincount(seg, weights=flat[:, d]) / counts
            share[d] = (counts * (seg_mean - mean[d]) ** 2).sum() / flat.shape[0] / safe_var[d]
        return share

    goal_share = segment_share(segments)
    # Null: the goal segments of the neighbouring environment (same lengths, not aligned with this signal).
    goal_share_null = segment_share(np.roll(segments, 1, axis=1))
    kurt = ((flat - mean) ** 4).mean(axis=0) / safe_var ** 2 - 3.0

    def num(x):
        return None if not np.isfinite(x) else float(x)

    out = []
    for d in range(dims):
        a = acf[:, d]
        out.append({
            "variance": float(var[d]),
            "autocorrelation": {str(lag): num(a[j]) for j, lag in enumerate(lags)},
            "lag_below_0.5": crossing(lags, a, 0.5) if np.isfinite(a).all() else None,
            "lag_below_1_over_e": crossing(lags, a, float(np.exp(-1.0))) if np.isfinite(a).all() else None,
            "integrated_autocorrelation_steps": num(sum((y0 + y1) / 2 * (x1 - x0) for x0, x1, y0, y1 in
                                                        zip([0, *lags], lags, [1.0, *a], a))),
            "hold_error": {str(h): num(hold_error[j, d]) for j, h in enumerate(holds)},
            "mean_abs_step_over_sd": num(step[d]),
            "sign_switch_rate": float(sign_switch[d]),
            "goal_segment_share": num(goal_share[d]),
            "goal_segment_share_null": num(goal_share_null[d]),
            "excess_kurtosis": num(kurt[d]),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--pca_basis", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lags", default="1,2,3,4,6,8,12,20,30,60")
    parser.add_argument("--holds", default="1,2,3,6,12,30")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    lags = [int(v) for v in args.lags.split(",")]
    holds = [int(v) for v in args.holds.split(",")]
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"Output or partial directory already exists: {output_dir}")

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_envs, vector_steps = int(manifest["num_envs"]), int(manifest["vector_steps"])
    shape = (vector_steps, num_envs)
    observations = np.load(dataset_dir / "observations.npy", mmap_mode="r")
    actions = np.asarray(np.load(dataset_dir / "actions.npy", mmap_mode="r"), dtype=np.float64)
    env_ids = np.load(dataset_dir / "env_ids.npy", mmap_mode="r")
    if observations.shape != (num_envs * vector_steps, OBS_DIM) or actions.shape != (num_envs * vector_steps, ACTION_DIM):
        raise ValueError(f"Unexpected shapes: obs {observations.shape}, actions {actions.shape}")
    if not np.array_equal(env_ids[:num_envs], np.arange(num_envs)):
        raise ValueError("Rows are not ordered vector-step-major with env IDs 0..num_envs-1")
    episode_ids = np.asarray(np.load(dataset_dir / "episode_ids.npy"), dtype=np.int64).reshape(shape)
    episode_steps = np.asarray(np.load(dataset_dir / "episode_steps.npy"), dtype=np.int64).reshape(shape)

    basis = np.load(args.pca_basis)
    mu, w = basis["mu_robot"].astype(np.float64), basis["W_robot"].astype(np.float64)
    eigenvalues = basis["eigenvalues_robot"].astype(np.float64)
    names = [str(n).replace("robot0_", "") for n in basis["action_names"]]
    failures: list[str] = []

    # Goal segments: a new segment starts at a goal change or at an episode change of the environment.
    goal = np.asarray(observations[:, GOAL_ROT], dtype=np.float64).reshape(*shape, 4)
    new_segment = np.zeros(shape, dtype=bool)
    new_segment[0] = True
    new_segment[1:] = (quat_angle(goal[1:], goal[:-1]) > GOAL_CHANGE_ANGLE) | (episode_ids[1:] != episode_ids[:-1])
    per_env = np.cumsum(new_segment, axis=0) - 1
    offsets = np.concatenate([[0], np.cumsum(per_env[-1] + 1)[:-1]])
    segments = (per_env + offsets[None, :]).astype(np.int64)
    num_segments = int(segments.max()) + 1
    del goal

    actions = actions.reshape(*shape, ACTION_DIM)
    scores = (actions - mu) @ w
    pc = signal_stats(scores, episode_ids, episode_steps, segments, lags, holds)
    joints = signal_stats(actions, episode_ids, episode_steps, segments, lags, holds)
    for i, e in enumerate(pc):
        e["component"] = i + 1
        e["explained_variance_ratio"] = float(eigenvalues[i] / eigenvalues.sum())
        e["top_loadings"] = [[names[j], float(w[j, i])] for j in np.argsort(-np.abs(w[:, i]))[:4]]
        if abs(e["variance"] - eigenvalues[i]) > 1e-6 * max(1.0, eigenvalues[i]):
            failures.append(f"pc{i + 1}_variance")
        if e["hold_error"]["1"] not in (0.0, None):
            failures.append(f"pc{i + 1}_hold1")
    for j, e in enumerate(joints):
        e["joint"] = names[j]

    # hand-k as a whole: variance-weighted mean of its components (equals the multi-dimensional definition).
    hand = {}
    for k in HAND_KS:
        weight = eigenvalues[:k] / eigenvalues[:k].sum()
        hand[f"hand-{k}"] = {
            "autocorrelation": {str(lag): float(sum(weight[i] * pc[i]["autocorrelation"][str(lag)] for i in range(k)))
                                for lag in lags},
            "hold_error": {str(h): float(sum(weight[i] * pc[i]["hold_error"][str(h)] for i in range(k))) for h in holds},
            "goal_segment_share": float(sum(weight[i] * pc[i]["goal_segment_share"] for i in range(k))),
            "goal_segment_share_null": float(sum(weight[i] * pc[i]["goal_segment_share_null"] for i in range(k))),
        }

    # Slow feature analysis on the non-degenerate PCA subspace: whiten, then eigen-decompose the covariance of
    # the one-step difference. slowness = var(dz) / (2 var(z)) = 1 - lag-1 autocorrelation (stationary signal).
    keep = eigenvalues > 1e-6
    white = scores[..., keep] / np.sqrt(eigenvalues[keep])
    same = episode_ids[:-1] == episode_ids[1:]
    diff = (white[1:] - white[:-1])[same]
    slow_vals, slow_vecs = np.linalg.eigh(diff.T @ diff / diff.shape[0])
    direction = w[:, keep] @ (slow_vecs / np.sqrt(eigenvalues[keep])[:, None])  # joint-space weights, z has var 1
    slow = []
    for i in range(slow_vals.size):
        back = w[:, keep] @ (slow_vecs[:, i] * np.sqrt(eigenvalues[keep]))  # action change per unit of the feature
        slow.append({
            "rank": i + 1, "lag1_autocorrelation": float(1.0 - slow_vals[i] / 2.0),
            "share_of_action_variance": float((back ** 2).sum() / eigenvalues.sum()),
            "top_weights": [[names[j], float(direction[j, i])] for j in np.argsort(-np.abs(direction[:, i]))[:4]],
        })

    summary = {
        "schema_version": 1, "record_type": "shadow_pc_timescales",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir), "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "task": manifest.get("task"),
        "pca_basis": str(Path(args.pca_basis).resolve()), "pca_basis_sha256": sha256(Path(args.pca_basis)),
        "control_hz": CONTROL_HZ, "lags": lags, "holds": holds, "num_samples": int(episode_ids.size),
        "num_goal_segments": num_segments, "mean_goal_segment_steps": float(episode_ids.size / num_segments),
        "goal_segment_share_white_noise": float(num_segments / episode_ids.size),
        "hold_rule": "refresh when episode_step % N == 0, otherwise keep the previous value",
        "components": pc, "joints": joints, "hand_k": hand, "slow_features": slow, "failures": failures,
    }

    partial_dir.mkdir(parents=True)
    write_json(partial_dir / "summary.json", summary)
    header = ["signal", "variance", *[f"acf_lag{lag}" for lag in lags], "lag_below_0.5", "lag_below_1_over_e",
              "integrated_autocorrelation_steps", *[f"hold_error_{h}" for h in holds], "mean_abs_step_over_sd",
              "sign_switch_rate", "goal_segment_share", "goal_segment_share_null", "excess_kurtosis"]
    rows = [",".join(header)]
    for label, e in [(f"PC{e['component']}", e) for e in pc] + [(e["joint"], e) for e in joints]:
        vals = [e["variance"], *[e["autocorrelation"][str(lag)] for lag in lags], e["lag_below_0.5"],
                e["lag_below_1_over_e"], e["integrated_autocorrelation_steps"],
                *[e["hold_error"][str(h)] for h in holds], e["mean_abs_step_over_sd"], e["sign_switch_rate"],
                e["goal_segment_share"], e["goal_segment_share_null"], e["excess_kurtosis"]]
        rows.append(",".join([label, *("" if v is None else f"{v:.6f}" for v in vals)]))
    (partial_dir / "pc_timescales.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if not args.skip_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        for i in range(8):
            axes[0].plot([0, *lags], [1.0, *[pc[i]["autocorrelation"][str(lag)] for lag in lags]], marker="o", ms=3,
                         color=cmap(i / 7), label=f"PC{i + 1}")
        axes[0].axhline(0.0, color="grey", lw=0.8)
        axes[0].set(xscale="symlog", xlabel="lag (control steps, 60 Hz)", ylabel="autocorrelation",
                    title="Autocorrelation of PC 1-8")
        axes[0].set_xlim(0, max(lags))
        axes[0].legend(fontsize=8, ncol=2)
        x = np.arange(1, ACTION_DIM + 1)
        axes[1].bar(x - 0.2, [e["autocorrelation"]["1"] or 0 for e in pc], width=0.4, label="lag 1")
        axes[1].bar(x + 0.2, [e["autocorrelation"]["6"] or 0 for e in pc], width=0.4, label="lag 6 (10 Hz)")
        for k in HAND_KS:
            axes[1].axvline(k + 0.5, color="grey", ls=":", lw=1)
        axes[1].set(xlabel="principal component (dotted: hand-2 / 4 / 8)", ylabel="autocorrelation",
                    title="Each PC: lag 1 and lag 6", xticks=x)
        axes[1].legend(fontsize=8)
        axes[2].bar(x - 0.2, [e["goal_segment_share"] or 0 for e in pc], width=0.4, label="own goal segments")
        axes[2].bar(x + 0.2, [e["goal_segment_share_null"] or 0 for e in pc], width=0.4, color="lightgrey",
                    label="another env's segments (null)")
        for k in HAND_KS:
            axes[2].axvline(k + 0.5, color="grey", ls=":", lw=1)
        axes[2].set(xlabel="principal component", ylabel="share of the PC's variance",
                    title="Variance explained by the per-goal mean", xticks=x)
        axes[2].legend(fontsize=8)
        fig.suptitle(f"Time structure of action PCs, {episode_ids.size:,} expert transitions ({dataset_dir.name})",
                     fontsize=10)
        fig.savefig(partial_dir / "pc_timescales.png", dpi=150)
        plt.close(fig)

    lines = [f"{sha256(p)}  {p.name}" for p in sorted(partial_dir.iterdir()) if p.name != "SHA256SUMS"]
    (partial_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial_dir.rename(output_dir)

    for e in pc[:8]:
        print(f"PC{e['component']} var_ratio={e['explained_variance_ratio']:.3f} acf1={e['autocorrelation']['1']:.3f} "
              f"acf6={e['autocorrelation']['6']:.3f} hold6={e['hold_error']['6']:.3f} "
              f"goal_share={e['goal_segment_share']:.3f} (null {e['goal_segment_share_null']:.3f}) sign_switch={e['sign_switch_rate']:.3f}")
    print(f"slowest_direction_acf1={slow[0]['lag1_autocorrelation']:.3f} "
          f"variance_share={slow[0]['share_of_action_variance']:.3f}")
    for msg in failures:
        print("check_failure:", msg)
    print(f"pc_timescales_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
