#!/usr/bin/env python3
"""Time structure of the expert's measured joint positions, next to that of its action commands (CPU).

Question: the action commands (joint position targets) forget themselves in about one control step
(analyze_pc_timescales.py). Is that also true of the hand's motion, or only of the commands?

Signals, all from the recorded expert rollout, within episodes, same statistics as analyze_pc_timescales.py
(autocorrelation, hold error, goal-segment share with an unaligned null, step change):
* joints: the 24 measured joint positions (observation slice hand_dof_pos, normalised by the joint limits);
* posture PCs: principal components of those 24 positions (own PCA, computed here);
* command directions: the positions of the 20 actuated joints projected on the action PCA basis, i.e. the
  posture along the same directions that hand-k sends as commands;
* slow features: the slowest linear directions of the joint positions.
The 20 actuated joints are found from the data: each command is matched to the position it correlates with
best one to COMMAND_LAGS steps later (a position target is followed by its joint). The match must be one
to one; it is reported with its correlation.
Diagnostic only: no basis from this script is used by any intent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from analyze_intent_hold import GOAL_CHANGE_ANGLE, quat_angle, write_json
from analyze_pc_timescales import signal_stats
from intent_modules import GOAL_ROT, OBS_DIM, sha256

ACTION_DIM = 20
CONTROL_HZ = 60.0
COMMAND_LAGS = (1, 2, 3, 4, 6)
MIN_STD = 1e-3  # a signal with a smaller standard deviation is treated as constant


def goal_segments(goal: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """Segment label per (step, env): a new segment at a goal change or an episode change."""
    new = np.zeros(episode_ids.shape, dtype=bool)
    new[0] = True
    new[1:] = (quat_angle(goal[1:], goal[:-1]) > GOAL_CHANGE_ANGLE) | (episode_ids[1:] != episode_ids[:-1])
    per_env = np.cumsum(new, axis=0) - 1
    offsets = np.concatenate([[0], np.cumsum(per_env[-1] + 1)[:-1]])
    return (per_env + offsets[None, :]).astype(np.int64)


def lagged_correlation(a: np.ndarray, p: np.ndarray, episode_ids: np.ndarray, lag: int) -> np.ndarray:
    """corr(a_t[j], p_{t+lag}[i]) within episodes -> [dims of a, dims of p]; NaN for constant signals."""
    same = episode_ids[:-lag] == episode_ids[lag:] if lag else np.ones(episode_ids.shape, dtype=bool)
    x = (a[:-lag] if lag else a)[same]
    y = (p[lag:] if lag else p)[same]
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    sx, sy = x.std(axis=0), y.std(axis=0)
    sx[sx < MIN_STD] = np.nan
    sy[sy < MIN_STD] = np.nan
    return (x.T @ y) / x.shape[0] / np.outer(sx, sy)


def slow_directions(z: np.ndarray, episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slow feature analysis on the non-constant subspace. Returns lag-1 autocorrelations (slowest first),
    the share of total variance each feature explains, and the weights [dims, features] (unit-variance features)."""
    flat = z.reshape(-1, z.shape[-1])
    mean = flat.mean(axis=0)
    cov = np.cov(flat - mean, rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > MIN_STD ** 2
    vals, vecs = vals[keep], vecs[:, keep]
    white = ((z - mean) @ vecs) / np.sqrt(vals)
    same = episode_ids[:-1] == episode_ids[1:]
    diff = (white[1:] - white[:-1])[same]
    slow_vals, slow_vecs = np.linalg.eigh(diff.T @ diff / diff.shape[0])
    weights = vecs @ (slow_vecs / np.sqrt(vals)[:, None])
    back = vecs @ (slow_vecs * np.sqrt(vals)[:, None])
    return 1.0 - slow_vals / 2.0, (back ** 2).sum(axis=0) / cov.trace(), weights


def label(stats: list[dict], key: str, names: list) -> list[dict]:
    for e, n in zip(stats, names):
        e[key] = n
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--pca_basis", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lags", default="1,2,3,4,6,8,12,20,30,60,120")
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
    start, stop = manifest.get("observation_slices", {}).get("hand_dof_pos", [0, 24])
    observations = np.load(dataset_dir / "observations.npy", mmap_mode="r")
    env_ids = np.load(dataset_dir / "env_ids.npy", mmap_mode="r")
    if observations.shape != (num_envs * vector_steps, OBS_DIM):
        raise ValueError(f"Unexpected observation shape {observations.shape}")
    if not np.array_equal(env_ids[:num_envs], np.arange(num_envs)):
        raise ValueError("Rows are not ordered vector-step-major with env IDs 0..num_envs-1")
    episode_ids = np.asarray(np.load(dataset_dir / "episode_ids.npy"), dtype=np.int64).reshape(shape)
    episode_steps = np.asarray(np.load(dataset_dir / "episode_steps.npy"), dtype=np.int64).reshape(shape)
    positions = np.asarray(observations[:, start:stop], dtype=np.float64).reshape(*shape, stop - start)
    actions = np.asarray(np.load(dataset_dir / "actions.npy", mmap_mode="r"), dtype=np.float64).reshape(*shape, ACTION_DIM)
    segments = goal_segments(np.asarray(observations[:, GOAL_ROT], dtype=np.float64).reshape(*shape, 4), episode_ids)
    num_dofs = positions.shape[-1]
    failures: list[str] = []

    basis = np.load(args.pca_basis)
    w_action = basis["W_robot"].astype(np.float64)
    action_eigenvalues = basis["eigenvalues_robot"].astype(np.float64)
    action_names = [str(n).replace("robot0_", "") for n in basis["action_names"]]

    # Match each command to the joint that follows it.
    corr = np.stack([lagged_correlation(actions, positions, episode_ids, lag) for lag in COMMAND_LAGS])
    best = np.nanmax(corr, axis=0)  # [actions, dofs]
    matching = []
    used: dict[int, str] = {}
    for j, name in enumerate(action_names):
        if np.isnan(best[j]).all():
            matching.append({"action": name, "dof": None, "correlation": None, "note": "constant command"})
            continue
        order = np.argsort(-np.nan_to_num(best[j], nan=-np.inf))
        i = int(order[0])
        matching.append({"action": name, "dof": i, "correlation": float(best[j, i]),
                         "best_lag": int(COMMAND_LAGS[int(np.nanargmax(corr[:, j, i]))]),
                         "runner_up_dof": int(order[1]), "runner_up_correlation": float(best[j, order[1]])})
        if i in used:
            failures.append(f"dof {i} matched by {used[i]} and {name}")
        used[i] = name
    dof_names = [used.get(i, f"dof{i}") for i in range(num_dofs)]
    matched = [m for m in matching if m["dof"] is not None]

    # Joints, posture PCs, slow features.
    joints = label(signal_stats(positions, episode_ids, episode_steps, segments, lags, holds), "joint", dof_names)
    flat = positions.reshape(-1, num_dofs)
    mean = flat.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(flat - mean, rowvar=False, bias=True))
    eigenvalues, vectors = eigenvalues[::-1].clip(min=0.0), vectors[:, ::-1]
    posture = signal_stats((positions - mean) @ vectors, episode_ids, episode_steps, segments, lags, holds)
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    for i, e in enumerate(posture):
        e["component"] = i + 1
        e["explained_variance_ratio"] = float(eigenvalues[i] / eigenvalues.sum())
        e["cumulative_explained_variance"] = float(cumulative[i])
        e["top_loadings"] = [[dof_names[j], float(vectors[j, i])] for j in np.argsort(-np.abs(vectors[:, i]))[:4]]
    slow_acf, slow_share, slow_weights = slow_directions(positions, episode_ids)
    slow = [{"rank": i + 1, "lag1_autocorrelation": float(slow_acf[i]), "share_of_position_variance": float(slow_share[i]),
             "top_weights": [[dof_names[j], float(slow_weights[j, i])] for j in np.argsort(-np.abs(slow_weights[:, i]))[:4]]}
            for i in range(slow_acf.size)]

    # Posture along the action PCA directions (only if every non-constant command found its joint).
    command_directions = None
    if not failures:
        cols = [m["dof"] for m in matching]
        sub = np.zeros((*shape, ACTION_DIM))
        for j, c in enumerate(cols):
            if c is not None:
                sub[..., j] = positions[..., c] - mean[c]
        command_directions = signal_stats(sub @ w_action, episode_ids, episode_steps, segments, lags, holds)
        total = sum(e["variance"] for e in command_directions)
        for i, e in enumerate(command_directions):
            e["component"] = i + 1
            e["command_explained_variance_ratio"] = float(action_eigenvalues[i] / action_eigenvalues.sum())
            e["position_variance_share"] = float(e["variance"] / total)

    # The commands themselves, for the same lags (reference in the same file).
    commands = signal_stats((actions - basis["mu_robot"].astype(np.float64)) @ w_action, episode_ids, episode_steps,
                            segments, lags, holds)
    for i, e in enumerate(commands):
        e["component"] = i + 1
    same = episode_ids[:-1] == episode_ids[1:]
    sd_p, sd_a = flat.std(axis=0), actions.reshape(-1, ACTION_DIM).std(axis=0)
    behaviour = {
        "position_mean_abs_step_change": float(np.abs(positions[1:] - positions[:-1])[same].mean()),
        "command_mean_abs_step_change": float(np.abs(actions[1:] - actions[:-1])[same].mean()),
        "position_sd_mean": float(sd_p.mean()), "command_sd_mean": float(sd_a.mean()),
        "position_tracking_error_rms_by_lag": {},
    }
    if not failures:
        for lag in (0, *COMMAND_LAGS):
            ok = episode_ids[:-lag] == episode_ids[lag:] if lag else np.ones(shape, dtype=bool)
            err = [((positions[lag:] if lag else positions)[..., m["dof"]] -
                    (actions[:-lag] if lag else actions)[..., action_names.index(m["action"])])[ok] for m in matched]
            behaviour["position_tracking_error_rms_by_lag"][str(lag)] = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))

    summary = {
        "schema_version": 1, "record_type": "shadow_joint_timescales",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir), "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "task": manifest.get("task"),
        "pca_basis": str(Path(args.pca_basis).resolve()), "pca_basis_sha256": sha256(Path(args.pca_basis)),
        "hand_dof_pos_slice": [int(start), int(stop)], "units": "positions and commands normalised to [-1, 1] by joint limits",
        "control_hz": CONTROL_HZ, "lags": lags, "holds": holds, "num_samples": int(episode_ids.size),
        "num_goal_segments": int(segments.max()) + 1, "mean_goal_segment_steps": float(episode_ids.size / (segments.max() + 1)),
        "command_lags_for_matching": list(COMMAND_LAGS), "matching": matching, "behaviour": behaviour,
        "joints": joints, "posture_components": posture, "command_directions": command_directions,
        "command_components": commands, "slow_features": slow, "failures": failures,
    }

    partial_dir.mkdir(parents=True)
    write_json(partial_dir / "summary.json", summary)
    header = ["group", "signal", "variance", *[f"acf_lag{lag}" for lag in lags], "lag_below_0.5", "lag_below_1_over_e",
              *[f"hold_error_{h}" for h in holds], "goal_segment_share", "goal_segment_share_null"]
    rows = [",".join(header)]
    groups = [("joint", [(e["joint"], e) for e in joints]), ("posture_pc", [(f"PC{e['component']}", e) for e in posture]),
              ("command_pc", [(f"PC{e['component']}", e) for e in commands])]
    if command_directions:
        groups.append(("position_on_command_pc", [(f"PC{e['component']}", e) for e in command_directions]))
    for group, entries in groups:
        for name, e in entries:
            vals = [e["variance"], *[e["autocorrelation"][str(lag)] for lag in lags], e["lag_below_0.5"],
                    e["lag_below_1_over_e"], *[e["hold_error"][str(h)] for h in holds], e["goal_segment_share"],
                    e["goal_segment_share_null"]]
            rows.append(",".join([group, name, *("" if v is None else f"{v:.6f}" for v in vals)]))
    (partial_dir / "joint_timescales.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if not args.skip_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        xs = [0, *lags]
        for i in range(8):
            axes[0].plot(xs, [1.0, *[posture[i]["autocorrelation"][str(lag)] for lag in lags]], marker="o", ms=3,
                         color=cmap(i / 7), label=f"posture PC{i + 1}")
        for i in range(8):
            axes[0].plot(xs, [1.0, *[commands[i]["autocorrelation"][str(lag)] for lag in lags]], color="grey", lw=0.8,
                         ls="--", label="command PC1-8" if i == 0 else None)
        axes[0].axhline(0.0, color="grey", lw=0.8)
        axes[0].set(xscale="symlog", xlim=(0, max(lags)), xlabel="lag (control steps, 60 Hz)", ylabel="autocorrelation",
                    title="Posture PCs (colour) vs command PCs (dashed)")
        axes[0].legend(fontsize=7, ncol=2)
        x = np.arange(1, 9)
        for off, (entries, name, color) in enumerate(((commands, "command PC", "grey"), (posture, "posture PC", "C0"))):
            for h, alpha in ((6, 1.0), (30, 0.45)):
                axes[1].bar(x + (off - 0.5) * 0.4 + (0.1 if h == 30 else -0.1), [e["hold_error"][str(h)] for e in entries[:8]],
                            width=0.2, color=color, alpha=alpha, label=f"{name}, hold {h}")
        axes[1].axhline(1.0, color="grey", ls=":", lw=1)
        axes[1].set(xlabel="principal component", ylabel="hold error / variance (1 = the mean)",
                    title="Hold error at 10 Hz and 2 Hz", xticks=x)
        axes[1].legend(fontsize=7)
        x = np.arange(1, 13)
        axes[2].bar(x - 0.2, [e["goal_segment_share"] or 0 for e in posture[:12]], width=0.4, label="own goal segments")
        axes[2].bar(x + 0.2, [e["goal_segment_share_null"] or 0 for e in posture[:12]], width=0.4, color="lightgrey",
                    label="another env's segments (null)")
        axes[2].set(xlabel="posture principal component", ylabel="share of the PC's variance",
                    title="Variance explained by the per-goal mean", xticks=x)
        axes[2].legend(fontsize=8)
        fig.suptitle(f"Time structure of joint positions, {episode_ids.size:,} expert transitions ({dataset_dir.name})",
                     fontsize=10)
        fig.savefig(partial_dir / "joint_timescales.png", dpi=150)
        plt.close(fig)

    lines = [f"{sha256(p)}  {p.name}" for p in sorted(partial_dir.iterdir()) if p.name != "SHA256SUMS"]
    (partial_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial_dir.rename(output_dir)

    print("matching:", " ".join(f"{m['action']}->{m['dof']}({m['correlation']:.2f})" if m["dof"] is not None
                                else f"{m['action']}->none" for m in matching))
    for e in posture[:8]:
        print(f"posture PC{e['component']} var_ratio={e['explained_variance_ratio']:.3f} acf1={e['autocorrelation']['1']:.3f} "
              f"acf6={e['autocorrelation']['6']:.3f} acf30={e['autocorrelation']['30']:.3f} hold6={e['hold_error']['6']:.3f} "
              f"hold30={e['hold_error']['30']:.3f} goal_share={e['goal_segment_share']:.3f} (null {e['goal_segment_share_null']:.3f})")
    print(f"slowest_position_direction_acf1={slow[0]['lag1_autocorrelation']:.4f} "
          f"variance_share={slow[0]['share_of_position_variance']:.3f}")
    for msg in failures:
        print("check_failure:", msg)
    print(f"joint_timescales_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
