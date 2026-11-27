#!/usr/bin/env python3
"""Fit the joint-position PCA basis used by the joint-K intent.

The basis is the PCA of the expert's measured positions of the 20 actuated joints, taken from
observation slice ``hand_dof_pos`` and reordered to the action vector's order. Actions are joint
targets scaled to [-1, 1] by the same joint limits that normalize ``hand_dof_pos``, so the intent
joint-K = W_joint[:, :K]^T (a_expert - mu_joint) expresses the expert's commanded posture in the
coordinates of its actual posture synergies.

Diagnostics written next to the basis:
* tracking: per action dim, how well the target error a_t - q_t predicts the motion q_{t+1} - q_t
  of its mapped joint versus every other actuated joint (the mapped joint should win if the index
  map is right);
* action variance captured: fraction of the expert action variance carried by the first K joint
  components, compared with the action-PCA basis itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from analyze_action_pca import (
    CHUNK_ROWS,
    SIGN_CONVENTION,
    basis_summary,
    chunked_moments,
    principal_components,
    read_checksums,
    sha256,
    write_json,
)


# Isaac Lab sorts actuated_dof_indices, so action i drives articulation joint ACTUATED_DOF_INDICES[i]
# (names in collect_shadow_rollout.ACTION_NAMES). The four passive distal joints 17, 18, 19, 22 are skipped.
ACTUATED_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 23]
NUM_DOFS = 24
CHECK_KS = [1, 2, 4, 8, 20]
MIN_ACTION_STD = 0.05  # near-constant actions (e.g. MFJ2 pinned at -1) carry no tracking signal


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)


def tracking_correlations(actions: np.ndarray, joints: np.ndarray, episode_ids: np.ndarray, num_envs: int) -> dict:
    """Check the action-to-joint map on within-episode consecutive pairs, rows ordered (step, env).

    Pass criterion (PD position control): the target error a_t[i] - q_t[j] should predict the motion
    q_{t+1}[j] - q_t[j] best for the mapped joint j among all actuated joints. corr(a_t[i], q_{t+1}[j])
    is also reported, but it is weak for the heavy, slow wrist joints and is not used to pass or fail.
    """
    steps = actions.shape[0] // num_envs
    a_rows, q_rows, n_rows = [], [], []
    for step in range(steps - 1):
        now = slice(step * num_envs, (step + 1) * num_envs)
        nxt = slice((step + 1) * num_envs, (step + 2) * num_envs)
        same = episode_ids[now] == episode_ids[nxt]
        a_rows.append(np.asarray(actions[now], dtype=np.float64)[same])
        q_rows.append(np.asarray(joints[now], dtype=np.float64)[same][:, ACTUATED_DOF_INDICES])
        n_rows.append(np.asarray(joints[nxt], dtype=np.float64)[same][:, ACTUATED_DOF_INDICES])
    a = np.concatenate(a_rows)
    q_now = np.concatenate(q_rows)
    q_next = np.concatenate(n_rows)
    action_std = a.std(0)
    moving = action_std >= MIN_ACTION_STD
    dims = a.shape[1]

    position_corr = _standardize(a).T @ _standardize(q_next) / a.shape[0]  # [action i, actuated joint j]
    error_corr = np.empty((dims, dims))
    for j in range(dims):
        motion = _standardize(q_next[:, j : j + 1] - q_now[:, j : j + 1])
        error_corr[:, j] = (_standardize(a - q_now[:, j : j + 1]) * motion).mean(0)

    def compare(corr: np.ndarray) -> dict:
        mapped = np.diag(corr).copy()
        others = corr.copy()
        np.fill_diagonal(others, -np.inf)
        return {
            "mapped_corr": mapped.tolist(),
            "best_other_actuated_corr": others.max(axis=1).tolist(),
            "best_other_actuated_joint": [ACTUATED_DOF_INDICES[j] for j in others.argmax(axis=1)],
            "mapped_is_best_for_all": bool(np.all((mapped > others.max(axis=1))[moving])),
        }

    error = compare(error_corr)
    return {
        "pairs": int(a.shape[0]),
        "action_std": action_std.tolist(),
        "checked_action_dims": np.nonzero(moving)[0].tolist(),
        "criterion": "corr(a_t[i] - q_t[j], q_{t+1}[j] - q_t[j]), mapped j must be best among actuated joints",
        "target_error_vs_motion": error,
        "position_next_step_diagnostic": compare(position_corr),
        "mapped_is_best_for_all": error["mapped_is_best_for_all"],
    }


def action_variance_captured(actions: np.ndarray, basis: np.ndarray, mu: np.ndarray) -> dict:
    """Fraction of total action variance in the span of the first K basis columns (centred on the action mean)."""
    action_mean, action_cov, _ = chunked_moments(actions, None)
    total = float(np.trace(action_cov))
    projected = basis.T @ action_cov @ basis
    captured = np.cumsum(np.diag(projected)) / total
    latent_mean = (action_mean - mu) @ basis
    return {
        "action_total_variance": total,
        "cumulative_fraction": captured.tolist(),
        "latent_std": np.sqrt(np.maximum(np.diag(projected), 0.0)).tolist(),
        "latent_mean": latent_mean.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--action_pca_basis", default=None, help="pca_basis.npz of the same rollout, for comparison")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    checksums = read_checksums(dataset_dir)
    start, stop = manifest["observation_slices"]["hand_dof_pos"]
    if stop - start != NUM_DOFS:
        raise ValueError(f"hand_dof_pos has {stop - start} DOFs, expected {NUM_DOFS}")
    num_envs = int(manifest["num_envs"])

    observations = np.load(dataset_dir / "observations.npy", mmap_mode="r")
    actions = np.load(dataset_dir / "actions.npy", mmap_mode="r")
    env_ids = np.load(dataset_dir / "env_ids.npy")
    episode_ids = np.load(dataset_dir / "episode_ids.npy")
    if not np.array_equal(env_ids.reshape(-1, num_envs), np.tile(np.arange(num_envs), (env_ids.size // num_envs, 1))):
        raise ValueError("Rollout rows are not ordered (step, env); tracking check needs that order")

    joints = np.empty((observations.shape[0], NUM_DOFS), dtype=np.float32)
    for row in range(0, observations.shape[0], CHUNK_ROWS):
        joints[row : row + CHUNK_ROWS] = observations[row : row + CHUNK_ROWS, start:stop]
    actuated = np.ascontiguousarray(joints[:, ACTUATED_DOF_INDICES])

    mu, covariance, count = chunked_moments(actuated, None)
    eigenvalues, basis = principal_components(covariance)
    summary_block = basis_summary(eigenvalues)

    tracking = tracking_correlations(actions, joints, episode_ids, num_envs)
    captured = action_variance_captured(actions, basis, mu)
    comparison = None
    if args.action_pca_basis:
        action_basis = np.load(args.action_pca_basis)
        action_eigen = action_basis["eigenvalues_robot"]
        comparison = {
            "action_pca_basis": str(Path(args.action_pca_basis).resolve()),
            "action_pca_basis_sha256": sha256(Path(args.action_pca_basis)),
            "action_pca_cumulative_fraction": (np.cumsum(action_eigen) / action_eigen.sum()).tolist(),
        }

    basis_path = output_dir / "joint_basis.npz"
    np.savez(
        basis_path,
        mu_joint=mu,
        W_joint=basis,
        eigenvalues_joint=eigenvalues,
        actuated_dof_indices=np.asarray(ACTUATED_DOF_INDICES),
    )
    summary = {
        "schema_version": 1,
        "record_type": "shadow_joint_position_intent_basis",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "observations_sha256_manifest": checksums.get("observations.npy"),
        "actions_sha256_manifest": checksums.get("actions.npy"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "task": manifest.get("task"),
        "num_samples": count,
        "units": "normalized to [-1, 1] by joint limits, same as the action vector",
        "actuated_dof_indices": ACTUATED_DOF_INDICES,
        "intent_definition": "joint-K = W_joint[:, :K]^T (a_expert - mu_joint)",
        "sign_convention": SIGN_CONVENTION,
        "joint20": summary_block,
        "tracking": tracking,
        "action_variance_captured_by_joint_basis": captured,
        "action_pca_comparison": comparison,
        "joint_basis_sha256": sha256(basis_path),
    }
    write_json(output_dir / "summary.json", summary)

    cumulative = summary_block["cumulative_explained_variance"]
    print(f"joint20 k2={cumulative[1]:.3f} k4={cumulative[3]:.3f} k8={cumulative[7]:.3f} "
          f"n95={summary_block['components_for_threshold']['0.95']}")
    print(f"tracking mapped_is_best_for_all={tracking['mapped_is_best_for_all']} "
          f"mapped_error_corr_min_checked="
          f"{min(tracking['target_error_vs_motion']['mapped_corr'][i] for i in tracking['checked_action_dims']):.3f}")
    for k in CHECK_KS:
        line = f"action_variance_captured k={k} joint_basis={captured['cumulative_fraction'][k - 1]:.3f}"
        if comparison:
            line += f" action_basis={comparison['action_pca_cumulative_fraction'][k - 1]:.3f}"
        print(line)
    if not tracking["mapped_is_best_for_all"]:
        print("joint_basis_status=FAIL (action-to-joint map does not track)")
        raise SystemExit(1)
    print("joint_basis_status=PASS")


if __name__ == "__main__":
    main()
