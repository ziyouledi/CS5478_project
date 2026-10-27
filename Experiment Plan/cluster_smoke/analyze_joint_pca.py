#!/usr/bin/env python3
"""PCA of the expert's measured joint positions, compared with the PCA of its action commands.

Both live in the same normalized units: actions are joint targets scaled to [-1, 1] by the joint
limits, and observation slice ``hand_dof_pos`` holds the joint positions unscaled the same way. The
question is whether the hand's actual motion is lower-dimensional than its (near bang-bang) commands.
Diagnostic only: no basis from this script is used by any intent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from analyze_action_pca import (
    CHUNK_ROWS,
    basis_summary,
    chunked_moments,
    principal_components,
    read_checksums,
    sha256,
    write_json,
)


WRIST_DOFS = 2  # hand_dof_pos starts with the two wrist joints, as the action vector does
LIMIT = 0.99  # |normalized value| at or above this counts as "at the joint limit"


def pca_block(name: str, array: np.ndarray, columns: slice) -> dict:
    data = array[:, columns] if columns != slice(None) else array
    mean, covariance, count = chunked_moments(data, None)
    eigenvalues, _ = principal_components(covariance)
    summary = basis_summary(eigenvalues)
    summary.update({"name": name, "dim": int(covariance.shape[0]), "samples": count, "mean": mean.tolist()})
    return summary


def at_limit_fraction(array: np.ndarray) -> float:
    hits = 0
    for start in range(0, array.shape[0], CHUNK_ROWS):
        hits += int((np.abs(np.asarray(array[start : start + CHUNK_ROWS])) >= LIMIT).sum())
    return hits / array.size


def step_changes(series: np.ndarray, episode_ids: np.ndarray, num_envs: int) -> dict:
    """Mean |x_t - x_{t-1}| within episodes, assuming rows are ordered step-major (step, env)."""
    steps = series.shape[0] // num_envs
    total = 0.0
    flips = 0
    pairs = 0
    previous = np.asarray(series[:num_envs], dtype=np.float64)
    previous_episode = episode_ids[:num_envs]
    for step in range(1, steps):
        rows = slice(step * num_envs, (step + 1) * num_envs)
        current = np.asarray(series[rows], dtype=np.float64)
        same = episode_ids[rows] == previous_episode
        delta = np.abs(current - previous)[same]
        total += float(delta.sum())
        flips += int((np.sign(current) != np.sign(previous))[same].sum())
        pairs += delta.size
        previous, previous_episode = current, episode_ids[rows]
    return {"mean_abs_step_change": total / pairs, "sign_change_fraction": flips / pairs, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    checksums = read_checksums(dataset_dir)
    start, stop = manifest["observation_slices"]["hand_dof_pos"]
    num_envs = int(manifest["num_envs"])

    observations = np.load(dataset_dir / "observations.npy", mmap_mode="r")
    actions = np.load(dataset_dir / "actions.npy", mmap_mode="r")
    env_ids = np.load(dataset_dir / "env_ids.npy")
    episode_ids = np.load(dataset_dir / "episode_ids.npy")
    positions = np.ascontiguousarray(observations[:, start:stop], dtype=np.float32)
    num_dofs = positions.shape[1]

    step_major = bool(np.array_equal(env_ids.reshape(-1, num_envs), np.tile(np.arange(num_envs), (env_ids.size // num_envs, 1))))

    blocks = {
        "joint_pos_all": pca_block(f"measured joint positions, all {num_dofs} DOFs", positions, slice(None)),
        "joint_pos_fingers": pca_block(f"measured joint positions, {num_dofs - WRIST_DOFS} finger DOFs",
                                       positions, slice(WRIST_DOFS, num_dofs)),
        "action_robot20": pca_block("action commands, 20-D", actions, slice(None)),
        "action_finger18": pca_block("action commands, 18-D fingers", actions, slice(WRIST_DOFS, actions.shape[1])),
    }
    behaviour = {
        "joint_pos": {"at_limit_fraction": at_limit_fraction(positions)},
        "action": {"at_limit_fraction": at_limit_fraction(actions)},
    }
    if step_major:
        behaviour["joint_pos"].update(step_changes(positions, episode_ids, num_envs))
        behaviour["action"].update(step_changes(actions, episode_ids, num_envs))

    summary = {
        "record_type": "shadow_expert_joint_position_pca",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "observations_sha256_manifest": checksums.get("observations.npy"),
        "actions_sha256_manifest": checksums.get("actions.npy"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "hand_dof_pos_slice": [start, stop],
        "units": "both normalized to [-1, 1] by joint limits",
        "at_limit_threshold": LIMIT,
        "rows_step_major": step_major,
        "notes": [
            "hand_dof_pos includes the passive distal joints coupled to the middle joints, which are not in the action vector.",
            "Wrist columns are assumed to be the first two DOFs, matching the action vector order.",
        ],
        "pca": blocks,
        "behaviour": behaviour,
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "cumulative_variance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["basis", "k", "cumulative"])
        for key, block in blocks.items():
            for k, value in enumerate(block["cumulative_explained_variance"], start=1):
                writer.writerow([key, k, f"{value:.10g}"])

    for key, block in blocks.items():
        cumulative = block["cumulative_explained_variance"]
        print(f"{key:18s} dim={block['dim']:2d} k2={cumulative[1]:.3f} k4={cumulative[3]:.3f} "
              f"k8={cumulative[7]:.3f} n95={block['components_for_threshold']['0.95']}")
    for key, values in behaviour.items():
        print(key, json.dumps(values))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for key, style in (("action_robot20", "-o"), ("action_finger18", "--o"), ("joint_pos_all", "-s"), ("joint_pos_fingers", "--s")):
        cumulative = blocks[key]["cumulative_explained_variance"]
        axis.plot(np.arange(1, len(cumulative) + 1), cumulative, style, ms=3, label=blocks[key]["name"])
    axis.plot([2], [0.8], "*", ms=12, color="tab:red", label="human grasps (Santello 1998)")
    axis.axhline(0.95, color="0.6", lw=0.8, ls="--")
    axis.set(xlabel="number of components k", ylabel="cumulative explained variance", ylim=(0, 1.02))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.savefig(output_dir / "joint_vs_action_pca.png", dpi=150)
    plt.close(fig)
    print("joint_pca_status=PASS")


if __name__ == "__main__":
    main()
