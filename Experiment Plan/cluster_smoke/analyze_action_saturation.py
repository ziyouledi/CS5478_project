#!/usr/bin/env python3
"""Measure how often deterministic expert actions sit at the [-1, 1] bounds and how often they switch."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ACTION_DIM = 20
CHUNK_STEPS = 256
HIST_BINS = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_saturation(path: Path, names: list[str], low: np.ndarray, high: np.ndarray, hist: np.ndarray,
                    edges: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    index = np.arange(len(names))
    axes[0].barh(index, low, color="#4C72B0", label="a ≤ -1 + tol")
    axes[0].barh(index, high, left=low, color="#DD8452", label="a ≥ 1 - tol")
    axes[0].set_yticks(index, [name.replace("robot0_", "") for name in names])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="fraction of transitions", xlim=(0, 1))
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    density = hist / hist.sum(axis=1, keepdims=True)
    image = axes[1].imshow(
        np.log10(density + 1e-6), aspect="auto", cmap="viridis",
        extent=(edges[0], edges[-1], len(names) - 0.5, -0.5),
    )
    axes[1].set_yticks(index, [name.replace("robot0_", "") for name in names])
    axes[1].set(xlabel="action value")
    fig.colorbar(image, ax=axes[1], label="log10 fraction per bin")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="An action counts as saturated when |a| >= 1 - tolerance.")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"Output or partial directory already exists: {output_dir}")

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest["action_names"]
    num_envs = int(manifest["num_envs"])
    vector_steps = int(manifest["vector_steps"])
    actions = np.load(dataset_dir / "actions.npy", mmap_mode="r")
    episode_ids = np.load(dataset_dir / "episode_ids.npy", mmap_mode="r")
    env_ids = np.load(dataset_dir / "env_ids.npy", mmap_mode="r")
    if actions.shape != (num_envs * vector_steps, ACTION_DIM):
        raise ValueError(f"Unexpected actions shape {actions.shape} for {num_envs} x {vector_steps}")
    if not np.array_equal(env_ids[:num_envs], np.arange(num_envs)):
        raise ValueError("Rows are not ordered vector-step-major with env IDs 0..num_envs-1")

    bound = 1.0 - args.tolerance
    edges = np.linspace(-1.0, 1.0, HIST_BINS + 1)
    count = 0
    low = np.zeros(ACTION_DIM, dtype=np.int64)
    high = np.zeros(ACTION_DIM, dtype=np.int64)
    hist = np.zeros((ACTION_DIM, HIST_BINS), dtype=np.int64)
    saturated_per_row = np.zeros(ACTION_DIM + 1, dtype=np.int64)
    pairs = 0
    sign_switch = np.zeros(ACTION_DIM, dtype=np.int64)
    both_saturated = np.zeros(ACTION_DIM, dtype=np.int64)
    saturated_flip = np.zeros(ACTION_DIM, dtype=np.int64)
    mean_abs_delta = np.zeros(ACTION_DIM, dtype=np.float64)

    previous_actions = None
    previous_episodes = None
    for first_step in range(0, vector_steps, CHUNK_STEPS):
        last_step = min(first_step + CHUNK_STEPS, vector_steps)
        rows = slice(first_step * num_envs, last_step * num_envs)
        block = np.asarray(actions[rows], dtype=np.float64).reshape(-1, num_envs, ACTION_DIM)
        episodes = np.asarray(episode_ids[rows]).reshape(-1, num_envs)

        flat = block.reshape(-1, ACTION_DIM)
        count += flat.shape[0]
        is_low = flat <= -bound
        is_high = flat >= bound
        low += is_low.sum(axis=0)
        high += is_high.sum(axis=0)
        saturated_per_row += np.bincount((is_low | is_high).sum(axis=1), minlength=ACTION_DIM + 1)
        for dim in range(ACTION_DIM):
            hist[dim] += np.histogram(flat[:, dim], bins=edges)[0]

        if previous_actions is not None:
            block = np.concatenate([previous_actions[None], block])
            episodes = np.concatenate([previous_episodes[None], episodes])
        current, following = block[:-1], block[1:]
        same_episode = (episodes[:-1] == episodes[1:])[..., None]
        pairs += int(same_episode.sum())
        sign_switch += ((np.sign(current) * np.sign(following) < 0) & same_episode).sum(axis=(0, 1))
        current_sat = np.abs(current) >= bound
        following_sat = np.abs(following) >= bound
        both = current_sat & following_sat & same_episode
        both_saturated += both.sum(axis=(0, 1))
        saturated_flip += (both & (np.sign(current) != np.sign(following))).sum(axis=(0, 1))
        mean_abs_delta += (np.abs(following - current) * same_episode).sum(axis=(0, 1))
        previous_actions = block[-1]
        previous_episodes = episodes[-1]

    low_fraction = low / count
    high_fraction = high / count
    per_dim = {
        name: {
            "fraction_low": float(low_fraction[i]),
            "fraction_high": float(high_fraction[i]),
            "fraction_saturated": float(low_fraction[i] + high_fraction[i]),
            "sign_switch_rate": float(sign_switch[i] / pairs),
            "saturated_flip_rate_given_both_saturated": (
                float(saturated_flip[i] / both_saturated[i]) if both_saturated[i] else None
            ),
            "mean_abs_step_change": float(mean_abs_delta[i] / pairs),
        }
        for i, name in enumerate(names)
    }
    overall_saturated = float((low.sum() + high.sum()) / (count * ACTION_DIM))
    summary = {
        "schema_version": 1,
        "record_type": "shadow_action_saturation",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "num_samples": count,
        "num_consecutive_pairs_within_episode": pairs,
        "tolerance": args.tolerance,
        "saturation_definition": f"|a| >= {bound}",
        "overall_fraction_saturated": overall_saturated,
        "mean_saturated_dims_per_transition": float(
            (np.arange(ACTION_DIM + 1) * saturated_per_row).sum() / count
        ),
        "saturated_dims_per_transition_histogram": saturated_per_row.tolist(),
        "histogram_bin_edges": edges.tolist(),
        "histogram_counts": {name: hist[i].tolist() for i, name in enumerate(names)},
        "per_dimension": per_dim,
    }

    partial_dir.mkdir(parents=True)
    write_json(partial_dir / "summary.json", summary)
    if not args.skip_plot:
        plot_saturation(
            partial_dir / "action_saturation.png", names, low_fraction, high_fraction, hist, edges,
            f"Expert action saturation (|a| ≥ {bound}), {count:,} transitions ({dataset_dir.name})",
        )
    checksum_lines = [
        f"{sha256(path)}  {path.name}" for path in sorted(partial_dir.iterdir()) if path.name != "SHA256SUMS"
    ]
    (partial_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.replace(partial_dir, output_dir)

    print(f"saturation_output_dir={output_dir}")
    print(f"saturation_samples={count}")
    print(f"saturation_overall_fraction={overall_saturated:.4f}")
    print(f"saturation_mean_dims_per_transition={summary['mean_saturated_dims_per_transition']:.2f}")
    for name, stats in per_dim.items():
        print(
            f"saturation_dim {name} sat={stats['fraction_saturated']:.3f} "
            f"low={stats['fraction_low']:.3f} high={stats['fraction_high']:.3f} "
            f"switch={stats['sign_switch_rate']:.3f} delta={stats['mean_abs_step_change']:.3f}"
        )
    print("saturation_status=PASS")


if __name__ == "__main__":
    main()
