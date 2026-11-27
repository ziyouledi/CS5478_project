#!/usr/bin/env python3
"""Compute the 20-D robot-action PCA basis and the 18-D finger PCA basis from an expert rollout."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ACTION_DIM = 20
WRIST_SLICE = slice(0, 2)
FINGER_SLICE = slice(2, 20)
CHUNK_ROWS = 262_144
ROBOT_CHECK_KS = [0, 1, 2, 4, 8, 20]
FINGER_CHECK_KS = [0, 1, 2, 4, 8, 18]
VARIANCE_THRESHOLDS = [0.8, 0.9, 0.95, 0.99]
SIGN_CONVENTION = (
    "Each principal component (a column of W) is flipped so that its entry with the largest "
    "absolute loading is positive; ties resolve to the lowest action index."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_checksums(dataset_dir: Path) -> dict[str, str]:
    checksums = {}
    for line in (dataset_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.strip()] = digest
    return checksums


def chunked_moments(actions: np.ndarray, clip: float | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Return float64 mean and population covariance, streaming over row chunks."""
    count = actions.shape[0]
    total = np.zeros(actions.shape[1], dtype=np.float64)
    for start in range(0, count, CHUNK_ROWS):
        block = np.asarray(actions[start : start + CHUNK_ROWS], dtype=np.float64)
        if clip is not None:
            np.clip(block, -clip, clip, out=block)
        total += block.sum(axis=0)
    mean = total / count

    scatter = np.zeros((actions.shape[1], actions.shape[1]), dtype=np.float64)
    for start in range(0, count, CHUNK_ROWS):
        block = np.asarray(actions[start : start + CHUNK_ROWS], dtype=np.float64)
        if clip is not None:
            np.clip(block, -clip, clip, out=block)
        block -= mean
        scatter += block.T @ block
    return mean, scatter / count, count


def principal_components(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    pivot = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[pivot, np.arange(basis.shape[1])])
    signs[signs == 0] = 1.0
    return eigenvalues, basis * signs


def empirical_checks(
    actions: np.ndarray,
    columns: slice,
    clip: float | None,
    mean: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    check_ks: list[int],
) -> dict:
    """Project the data onto the basis and compare with the eigen-decomposition."""
    count = actions.shape[0]
    dim = basis.shape[0]
    latent_sq = np.zeros(dim, dtype=np.float64)
    latent_cross = np.zeros((dim, dim), dtype=np.float64)
    residual_sq = {k: 0.0 for k in check_ks}
    for start in range(0, count, CHUNK_ROWS):
        block = np.asarray(actions[start : start + CHUNK_ROWS, columns], dtype=np.float64)
        if clip is not None:
            np.clip(block, -clip, clip, out=block)
        centered = block - mean
        latent = centered @ basis
        latent_sq += np.square(latent).sum(axis=0)
        latent_cross += latent.T @ latent
        for k in check_ks:
            reconstruction = latent[:, :k] @ basis[:, :k].T
            residual_sq[k] += float(np.square(centered - reconstruction).sum())

    latent_variance = latent_sq / count
    latent_cov = latent_cross / count
    off_diagonal = latent_cov - np.diag(np.diag(latent_cov))
    total_variance = float(eigenvalues.sum())
    return {
        "orthonormality_max_abs_error": float(np.max(np.abs(basis.T @ basis - np.eye(dim)))),
        "latent_variance_max_abs_error": float(np.max(np.abs(latent_variance - eigenvalues))),
        "latent_covariance_max_abs_off_diagonal": float(np.max(np.abs(off_diagonal))),
        "reconstruction_mse_per_dim": {
            str(k): residual_sq[k] / (count * dim) for k in check_ks
        },
        "predicted_reconstruction_mse_per_dim": {
            str(k): float(eigenvalues[k:].sum()) / dim for k in check_ks
        },
        "unexplained_variance_fraction": {
            str(k): float(eigenvalues[k:].sum()) / total_variance for k in check_ks
        },
    }


def basis_summary(eigenvalues: np.ndarray) -> dict:
    ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratio)
    return {
        "total_variance": float(eigenvalues.sum()),
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": ratio.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "components_for_threshold": {
            str(threshold): int(np.searchsorted(cumulative, threshold - 1e-12) + 1)
            for threshold in VARIANCE_THRESHOLDS
        },
    }


def write_variance_csv(path: Path, robot: dict, finger: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["basis", "k", "eigenvalue", "explained_variance_ratio", "cumulative"])
        for name, summary in (("robot20", robot), ("finger18", finger)):
            for index, (value, ratio, cumulative) in enumerate(
                zip(
                    summary["eigenvalues"],
                    summary["explained_variance_ratio"],
                    summary["cumulative_explained_variance"],
                ),
                start=1,
            ):
                writer.writerow([name, index, f"{value:.10g}", f"{ratio:.10g}", f"{cumulative:.10g}"])


def plot_variance(path: Path, robot: dict, finger: dict, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, summary, marker in (
        ("20-D robot (wrist + fingers)", robot, "o"),
        ("18-D fingers", finger, "s"),
    ):
        ks = np.arange(1, len(summary["eigenvalues"]) + 1)
        axes[0].plot(ks, summary["cumulative_explained_variance"], marker=marker, ms=4, label=name)
        axes[1].plot(ks, summary["explained_variance_ratio"], marker=marker, ms=4, label=name)
    for threshold in (0.9, 0.95):
        axes[0].axhline(threshold, color="0.6", lw=0.8, ls="--")
    for k in (2, 4, 8):
        axes[0].axvline(k, color="0.85", lw=0.8)
    axes[0].set(xlabel="number of components k", ylabel="cumulative explained variance", ylim=(0, 1.02))
    axes[1].set(xlabel="component index", ylabel="explained variance ratio", yscale="log")
    axes[0].set_xticks(range(0, 21, 2))
    axes[1].set_xticks(range(0, 21, 2))
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_samples", type=int, default=None)
    parser.add_argument(
        "--clip_actions",
        type=float,
        default=None,
        help="Clip actions to [-c, c] before PCA. Default: use stored actions unchanged.",
    )
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"Output or partial directory already exists: {output_dir}")

    manifest_path = dataset_dir / "manifest.json"
    actions_path = dataset_dir / "actions.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = read_checksums(dataset_dir)
    actions = np.load(actions_path, mmap_mode="r")

    action_names = manifest["action_names"]
    if len(action_names) != ACTION_DIM or not action_names[0].startswith("robot0_WRJ"):
        raise ValueError(f"Unexpected action_names in manifest: {action_names}")
    if not action_names[1].startswith("robot0_WRJ") or any("WRJ" in n for n in action_names[2:]):
        raise ValueError("Expected wrist joints at action indices 0:2 only")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Unexpected actions shape: {actions.shape}")
    if actions.shape[0] != manifest["total_samples"]:
        raise ValueError(f"actions rows {actions.shape[0]} != manifest {manifest['total_samples']}")
    if args.expected_samples is not None and actions.shape[0] != args.expected_samples:
        raise ValueError(f"actions rows {actions.shape[0]} != expected {args.expected_samples}")
    if "actions.npy" not in checksums:
        raise ValueError("actions.npy missing from dataset SHA256SUMS")

    finite = all(
        np.isfinite(actions[start : start + CHUNK_ROWS]).all()
        for start in range(0, actions.shape[0], CHUNK_ROWS)
    )
    if not finite:
        raise ValueError("actions.npy contains non-finite values")
    out_of_unit_range = sum(
        int(np.count_nonzero(np.abs(actions[start : start + CHUNK_ROWS]) > 1.0))
        for start in range(0, actions.shape[0], CHUNK_ROWS)
    )
    action_min = np.min(
        [actions[start : start + CHUNK_ROWS].min(axis=0) for start in range(0, actions.shape[0], CHUNK_ROWS)],
        axis=0,
    )
    action_max = np.max(
        [actions[start : start + CHUNK_ROWS].max(axis=0) for start in range(0, actions.shape[0], CHUNK_ROWS)],
        axis=0,
    )

    mean_robot, cov_robot, count = chunked_moments(actions, args.clip_actions)
    eig_robot, w_robot = principal_components(cov_robot)
    mean_finger = mean_robot[FINGER_SLICE].copy()
    eig_finger, w_finger = principal_components(cov_robot[FINGER_SLICE, FINGER_SLICE])

    robot_summary = basis_summary(eig_robot)
    finger_summary = basis_summary(eig_finger)
    robot_checks = empirical_checks(
        actions, slice(0, ACTION_DIM), args.clip_actions, mean_robot, w_robot, eig_robot, ROBOT_CHECK_KS
    )
    finger_checks = empirical_checks(
        actions, FINGER_SLICE, args.clip_actions, mean_finger, w_finger, eig_finger, FINGER_CHECK_KS
    )
    for name, checks in (("robot", robot_checks), ("finger", finger_checks)):
        if checks["orthonormality_max_abs_error"] > 1e-8:
            raise RuntimeError(f"{name} basis is not orthonormal: {checks}")
        if checks["latent_variance_max_abs_error"] > 1e-6:
            raise RuntimeError(f"{name} latent variance does not match eigenvalues: {checks}")

    partial_dir.mkdir(parents=True)
    np.savez(
        partial_dir / "pca_basis.npz",
        action_names=np.array(action_names),
        finger_action_names=np.array(action_names[FINGER_SLICE]),
        mu_robot=mean_robot,
        W_robot=w_robot,
        eigenvalues_robot=eig_robot,
        explained_variance_ratio_robot=np.array(robot_summary["explained_variance_ratio"]),
        mu_finger=mean_finger,
        W_finger=w_finger,
        eigenvalues_finger=eig_finger,
        explained_variance_ratio_finger=np.array(finger_summary["explained_variance_ratio"]),
    )
    write_variance_csv(partial_dir / "cumulative_variance.csv", robot_summary, finger_summary)
    if not args.skip_plot:
        plot_variance(
            partial_dir / "cumulative_variance.png",
            robot_summary,
            finger_summary,
            f"Expert action PCA, {count:,} transitions ({dataset_dir.name})",
        )

    summary = {
        "schema_version": 1,
        "record_type": "shadow_action_pca",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_actions_sha256": checksums["actions.npy"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "rollout_seed": manifest["seed"],
        "num_samples": count,
        "action_names": action_names,
        "wrist_indices": [0, 1],
        "finger_indices": list(range(2, ACTION_DIM)),
        "clip_actions": args.clip_actions,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "fraction_entries_outside_unit_range": out_of_unit_range / (count * ACTION_DIM),
        "covariance": "population (divide by N), float64",
        "basis_layout": "W[:, i] is principal component i, sorted by descending eigenvalue; "
        "intent z_k = W[:, :k].T @ (a - mu)",
        "sign_convention": SIGN_CONVENTION,
        "robot20": {**robot_summary, "checks": robot_checks},
        "finger18": {**finger_summary, "checks": finger_checks},
    }
    write_json(partial_dir / "summary.json", summary)

    checksum_lines = [
        f"{sha256(path)}  {path.name}" for path in sorted(partial_dir.iterdir()) if path.name != "SHA256SUMS"
    ]
    (partial_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.replace(partial_dir, output_dir)

    print(f"pca_output_dir={output_dir}")
    print(f"pca_samples={count}")
    print(f"pca_fraction_outside_unit_range={summary['fraction_entries_outside_unit_range']:.6f}")
    for name, block in (("robot20", summary["robot20"]), ("finger18", summary["finger18"])):
        cumulative = block["cumulative_explained_variance"]
        print(f"pca_{name}_cumulative_k2={cumulative[1]:.4f}")
        print(f"pca_{name}_cumulative_k4={cumulative[3]:.4f}")
        print(f"pca_{name}_cumulative_k8={cumulative[7]:.4f}")
        print(f"pca_{name}_components_for_0.95={block['components_for_threshold']['0.95']}")
    print("pca_status=PASS")


if __name__ == "__main__":
    main()
