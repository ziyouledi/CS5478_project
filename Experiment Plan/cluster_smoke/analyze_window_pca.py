#!/usr/bin/env python3
"""PCA over time windows of the expert's commands and joint positions (CPU).

Question: a per-step PCA says the hand needs about 14 dimensions and every component forgets itself within a
few steps. Is a stretch of motion (a "stride" of L control steps) nevertheless made of a few recurring
patterns, i.e. are windows of the trajectory low-dimensional or clustered?

For each signal (20 commands, 24 measured joint positions), each window length L and two ways of cutting
windows (sliding: consecutive non-overlapping windows; goal: windows starting at each goal change), within
episodes:
* PCA of the windows (L x D numbers each): components needed for 80 / 90 / 95 % of the variance,
  cumulative variance at fixed k, participation ratio;
* the same counts under two nulls built from the same data:
  - independent: time steps of a window unrelated (per-step spectra put side by side);
  - separable: every joint pattern shares one time course (spatial covariance x temporal correlation).
  Fewer components than the separable null means joint patterns that follow each other in a fixed order,
  beyond what smoothness alone gives;
* template share: variance of the mean window (the stereotyped response) over the total variance;
* k-means on the windows vs on Gaussian windows with the same covariance: a small dictionary of discrete
  patterns would explain more variance than the Gaussian null.
Diagnostic only: no basis from this script is used by any intent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from analyze_intent_hold import write_json
from analyze_joint_timescales import goal_segments
from intent_modules import GOAL_ROT, OBS_DIM, sha256

ACTION_DIM = 20
CONTROL_HZ = 60.0
LEVELS = (0.80, 0.90, 0.95)
SERIES = (("actual", "#2a78d6", "o", "-"), ("independent", "#eb6834", "s", "--"), ("separable", "#1baf7a", "^", ":"))


def window_starts(episode_ids: np.ndarray, segments: np.ndarray, length: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(t, env) of every window of `length` steps that stays inside one episode."""
    steps, envs = episode_ids.shape
    if mode == "sliding":
        t = np.arange(0, steps - length + 1, length)
        tt, ee = np.meshgrid(t, np.arange(envs), indexing="ij")
        tt, ee = tt.ravel(), ee.ravel()
    else:
        new = np.zeros(segments.shape, dtype=bool)
        new[0] = True
        new[1:] = segments[1:] != segments[:-1]
        tt, ee = np.nonzero(new)
        keep = tt <= steps - length
        tt, ee = tt[keep], ee[keep]
    keep = episode_ids[tt, ee] == episode_ids[tt + length - 1, ee]
    return tt[keep], ee[keep]


def gather(signal: np.ndarray, tt: np.ndarray, ee: np.ndarray, length: int) -> np.ndarray:
    """[windows, length, dims]."""
    return signal[tt[:, None] + np.arange(length)[None, :], ee[:, None]]


def count_for(values: np.ndarray, level: float) -> int:
    """Number of the largest values needed to reach `level` of their sum."""
    v = np.sort(np.clip(values, 0.0, None))[::-1]
    return int(np.searchsorted(np.cumsum(v) / v.sum(), level - 1e-12) + 1)


def spectrum_stats(values: np.ndarray, ks: list[int]) -> dict:
    v = np.sort(np.clip(values, 0.0, None))[::-1]
    cum = np.cumsum(v) / v.sum()
    return {
        **{f"n{int(round(level * 100))}": count_for(v, level) for level in LEVELS},
        "participation_ratio": float(v.sum() ** 2 / (v ** 2).sum()),
        "cumulative_variance": {str(k): float(cum[min(k, v.size) - 1]) for k in ks},
    }


def window_pca(x: np.ndarray, ks: list[int]) -> tuple[dict, np.ndarray, np.ndarray]:
    """x: [windows, length, dims]. Returns the statistics, the eigenvalues and eigenvectors of the window covariance."""
    n, length, dims = x.shape
    template = x.mean(axis=0)  # [length, dims]
    overall = template.mean(axis=0)  # [dims]
    flat = (x - template).reshape(n, length * dims)
    cov = flat.T @ flat / n
    values, vectors = np.linalg.eigh(cov)
    values, vectors = values[::-1].clip(min=0.0), vectors[:, ::-1]
    blocks = cov.reshape(length, dims, length, dims)
    spatial = [blocks[t, :, t, :] for t in range(length)]
    independent = np.concatenate([np.linalg.eigvalsh(s) for s in spatial])
    traces = np.array([[np.trace(blocks[a, :, b, :]) for b in range(length)] for a in range(length)])
    scale = np.sqrt(np.diag(traces))
    temporal = np.linalg.eigvalsh(traces / np.outer(scale, scale)).clip(min=0.0)
    separable = np.outer(np.linalg.eigvalsh(np.mean(spatial, axis=0)).clip(min=0.0), temporal).ravel()
    within = float(values.sum())
    between = float(((template - overall) ** 2).sum())
    stats = {
        "windows": int(n), "length": int(length), "dims": int(dims), "window_dims": int(length * dims),
        "template_share": between / (between + within),
        "actual": spectrum_stats(values, ks),
        "independent": spectrum_stats(independent, ks),
        "separable": spectrum_stats(separable, ks),
        "temporal_correlation_first_row": [float(v) for v in (traces / np.outer(scale, scale))[0]],
    }
    return stats, values, vectors


def kmeans_explained(flat: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 25) -> float:
    """Share of the variance of `flat` (already centred) explained by k centroids (Lloyd, random start)."""
    centres = flat[rng.choice(flat.shape[0], size=k, replace=False)].copy()
    norms = (flat ** 2).sum(axis=1)
    for _ in range(iterations):
        dist = norms[:, None] - 2.0 * flat @ centres.T + (centres ** 2).sum(axis=1)[None, :]
        assign = dist.argmin(axis=1)
        for j in range(k):
            members = flat[assign == j]
            centres[j] = members.mean(axis=0) if members.shape[0] else flat[rng.integers(flat.shape[0])]
    dist = norms[:, None] - 2.0 * flat @ centres.T + (centres ** 2).sum(axis=1)[None, :]
    return float(1.0 - dist.min(axis=1).clip(min=0.0).sum() / norms.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="1,3,6,12,21,30")
    parser.add_argument("--ks", default="2,4,8,16,32,64")
    parser.add_argument("--cluster_lengths", default="12,21")
    parser.add_argument("--cluster_ks", default="4,16,64")
    parser.add_argument("--cluster_windows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    lengths = [int(v) for v in args.lengths.split(",")]
    ks = [int(v) for v in args.ks.split(",")]
    cluster_lengths = [int(v) for v in args.cluster_lengths.split(",")]
    cluster_ks = [int(v) for v in args.cluster_ks.split(",")]
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
    positions = np.asarray(observations[:, start:stop], dtype=np.float64).reshape(*shape, stop - start)
    actions = np.asarray(np.load(dataset_dir / "actions.npy", mmap_mode="r"), dtype=np.float64).reshape(*shape, ACTION_DIM)
    segments = goal_segments(np.asarray(observations[:, GOAL_ROT], dtype=np.float64).reshape(*shape, 4), episode_ids)
    segment_steps = np.bincount(segments.ravel())
    signals = {"commands": actions, "positions": positions}
    rng = np.random.default_rng(args.seed)
    failures: list[str] = []

    results: list[dict] = []
    clusters: list[dict] = []
    for name, signal in signals.items():
        for mode in ("sliding", "goal"):
            for length in lengths:
                if mode == "goal" and length == 1:
                    continue
                tt, ee = window_starts(episode_ids, segments, length, mode)
                if tt.size < 10 * length * signal.shape[-1]:
                    failures.append(f"{name} {mode} L={length}: only {tt.size} windows")
                    continue
                x = gather(signal, tt, ee, length)
                stats, values, vectors = window_pca(x, ks)
                stats.update({"signal": name, "windows_cut": mode})
                if mode == "goal":
                    stats["share_of_windows_inside_one_goal"] = float((segment_steps[segments[tt, ee]] >= length).mean())
                if stats["actual"]["n95"] > stats["independent"]["n95"]:  # impossible: the spectrum majorises its blocks
                    failures.append(f"{name} {mode} L={length}: more components than the independent null")
                results.append(stats)
                print(f"{name:9s} {mode:7s} L={length:2d} windows={stats['windows']:7d} n95 actual={stats['actual']['n95']:4d} "
                      f"independent={stats['independent']['n95']:4d} separable={stats['separable']['n95']:4d} "
                      f"cum8={stats['actual']['cumulative_variance'].get('8', float('nan')):.3f} "
                      f"template={stats['template_share']:.3f}")
                if length in cluster_lengths:
                    pick = rng.choice(x.shape[0], size=min(args.cluster_windows, x.shape[0]), replace=False)
                    flat = (x[pick] - x.mean(axis=0)).reshape(pick.size, -1)
                    flat = flat - flat.mean(axis=0)
                    gauss = rng.standard_normal((pick.size, values.size)) * np.sqrt(values)[None, :] @ vectors.T
                    gauss = gauss - gauss.mean(axis=0)
                    for k in cluster_ks:
                        entry = {"signal": name, "windows_cut": mode, "length": length, "k": k, "windows": int(pick.size),
                                 "explained": kmeans_explained(flat, k, rng),
                                 "explained_gaussian_null": kmeans_explained(gauss, k, rng)}
                        clusters.append(entry)
                        print(f"  kmeans {name} {mode} L={length} k={k}: {entry['explained']:.3f} "
                              f"(gaussian null {entry['explained_gaussian_null']:.3f})")

    summary = {
        "schema_version": 1, "record_type": "shadow_window_pca",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir), "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "task": manifest.get("task"),
        "hand_dof_pos_slice": [int(start), int(stop)], "units": "positions and commands normalised to [-1, 1] by joint limits",
        "control_hz": CONTROL_HZ, "lengths": lengths, "ks": ks, "levels": list(LEVELS), "seed": args.seed,
        "num_samples": int(episode_ids.size), "num_goal_segments": int(segments.max()) + 1,
        "mean_goal_segment_steps": float(episode_ids.size / (segments.max() + 1)),
        "median_goal_segment_steps": float(np.median(segment_steps)),
        "windows": results, "clusters": clusters, "failures": failures,
    }

    partial_dir.mkdir(parents=True)
    write_json(partial_dir / "summary.json", summary)
    header = ["signal", "windows_cut", "length", "windows", "window_dims", "template_share",
              *[f"{s}_{n}" for s in ("actual", "independent", "separable") for n in ("n80", "n90", "n95")],
              "actual_participation_ratio", *[f"actual_cumvar_k{k}" for k in ks]]
    rows = [",".join(header)]
    for r in results:
        vals = [r["signal"], r["windows_cut"], r["length"], r["windows"], r["window_dims"], f"{r['template_share']:.6f}",
                *[r[s][n] for s in ("actual", "independent", "separable") for n in ("n80", "n90", "n95")],
                f"{r['actual']['participation_ratio']:.3f}", *[f"{r['actual']['cumulative_variance'][str(k)]:.6f}" for k in ks]]
        rows.append(",".join(str(v) for v in vals))
    (partial_dir / "window_pca.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if not args.skip_plot and results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True, facecolor="#fcfcfb")
        for ax, name in zip(axes[:2], signals):
            rows_ = [r for r in results if r["signal"] == name and r["windows_cut"] == "sliding"]
            xs = [r["length"] for r in rows_]
            for series, color, marker, style in SERIES:
                ax.plot(xs, [r[series]["n95"] for r in rows_], color=color, marker=marker, ms=6, lw=2, ls=style,
                        label={"actual": "windows of the rollout", "independent": "null: unrelated time steps",
                               "separable": "null: one shared time course"}[series])
            ax.set(xlabel="window length L (control steps, 60 Hz)", ylabel="components for 95 % of the variance",
                   title=f"{name}: sliding windows", xticks=xs)
            ax.legend(fontsize=8, frameon=False)
        rows_ = [c for c in clusters if c["windows_cut"] == "goal" and c["length"] == max(cluster_lengths)]
        labels = [f"{c['signal']}\nk = {c['k']}" for c in rows_]
        x = np.arange(len(rows_))
        axes[2].bar(x - 0.2, [c["explained"] for c in rows_], width=0.36, color="#2a78d6", label="windows of the rollout")
        axes[2].bar(x + 0.2, [c["explained_gaussian_null"] for c in rows_], width=0.36, color="#eb6834",
                    label="null: Gaussian, same covariance")
        axes[2].set(xticks=x, ylabel="variance explained by k cluster centres",
                    title=f"k-means, goal-aligned windows, L = {max(cluster_lengths)}")
        axes[2].set_xticklabels(labels, fontsize=8)
        axes[2].legend(fontsize=8, frameon=False)
        for ax in axes:
            ax.set_facecolor("#fcfcfb")
            ax.grid(axis="y", color=grid, lw=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=muted)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.title.set_color(ink)
        fig.suptitle(f"PCA over time windows, {episode_ids.size:,} expert transitions ({dataset_dir.name})", color=ink)
        fig.savefig(partial_dir / "window_pca.png", dpi=150)
        plt.close(fig)

    lines = [f"{sha256(p)}  {p.name}" for p in sorted(partial_dir.iterdir()) if p.name != "SHA256SUMS"]
    (partial_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial_dir.rename(output_dir)

    for msg in failures:
        print("check_failure:", msg)
    print(f"window_pca_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
