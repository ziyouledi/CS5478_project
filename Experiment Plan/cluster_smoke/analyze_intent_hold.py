#!/usr/bin/env python3
"""Offline zero-order-hold analysis of intent signals on a recorded expert rollout (CPU, no Isaac Sim).

For hold N the intent of an environment is refreshed when its episode step is a multiple of N and kept
otherwise. The script measures how stale each intent becomes: normalised hold error, autocorrelation,
the expert-action variance a held action intent still reconstructs, and the angle error of held
quaternion intents. The data come from the expert's own trajectories, so this is a property of the
signals, not the performance of a student trained under hold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import torch

from intent_modules import GOAL_ROT, OBS_DIM, WRIST, build_intent, canonical_quat, sha256


ACTION_DIM = 20
CONTROL_HZ = 60.0
HAND_KS = (2, 4, 8, 20)
QUAT_INTENTS = ("obj-4", "goal-4")
SUCCESS_TOLERANCE = 0.1  # rad, the environment's success tolerance
GOAL_CHANGE_ANGLE = 1e-2  # rad; float32 quaternions give up to ~1e-3 rad of noise, resampled goals far more
CHUNK_ROWS = 131072


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between unit quaternions, insensitive to the q / -q sign."""
    dot = np.abs((a * b).sum(axis=-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return 2.0 * np.arccos(dot.clip(max=1.0))


def held_index(episode_steps: np.ndarray, episode_ids: np.ndarray, hold: int) -> tuple[np.ndarray, np.ndarray]:
    """For [T, E] arrays: the vector step of the last refresh for every (t, env), and its validity."""
    steps_total = episode_steps.shape[0]
    time_index = np.arange(steps_total, dtype=np.int64)[:, None]
    refresh = np.where(episode_steps % hold == 0, time_index, -1)
    held = np.maximum.accumulate(refresh, axis=0)
    safe = np.maximum(held, 0)
    valid = (held >= 0) & (np.take_along_axis(episode_ids, safe, axis=0) == episode_ids)
    return safe, valid


def take(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, index[..., None], axis=0)


def compute_intents(
    observations: np.ndarray, actions: np.ndarray, basis, obj_lookahead: float, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    mu = basis["mu_robot"].astype(np.float64)
    w = basis["W_robot"].astype(np.float64)
    centered = actions - mu
    intents = {f"hand-{k}": centered @ w[:, :k] for k in HAND_KS}
    intents["wrist-2"] = actions[:, WRIST].copy()

    obj_modules = {name: build_intent(name, obj_lookahead=obj_lookahead) for name in ("obj-1", "obj-4")}
    parts: dict[str, list[np.ndarray]] = {"obj-1": [], "obj-4": [], "goal-4": []}
    for start in range(0, observations.shape[0], CHUNK_ROWS):
        obs = torch.from_numpy(np.asarray(observations[start : start + CHUNK_ROWS], dtype=np.float32))
        for name, module in obj_modules.items():
            parts[name].append(module(obs).double().numpy())
        parts["goal-4"].append(canonical_quat(obs[:, GOAL_ROT]).double().numpy())
    intents.update({name: np.concatenate(chunks) for name, chunks in parts.items()})
    return {name: value.reshape(*shape, -1) for name, value in intents.items()}


def reconstruct_action(name: str, held: np.ndarray, basis, action_mean: np.ndarray) -> np.ndarray:
    """Best linear guess of the 20-D expert action from a held action intent."""
    if name == "wrist-2":
        guess = np.broadcast_to(action_mean, held.shape[:-1] + (ACTION_DIM,)).copy()
        guess[..., WRIST] = held
        return guess
    k = held.shape[-1]
    return basis["mu_robot"].astype(np.float64) + held @ basis["W_robot"].astype(np.float64)[:, :k].T


def plot(path: Path, holds: list[int], results: dict, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = [CONTROL_HZ / n for n in holds]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, per_hold in results.items():
        axes[0].plot(rates, [per_hold[str(n)]["normalized_hold_error"] for n in holds], marker="o", label=name)
        if "action_variance_explained" in per_hold[str(holds[0])]:
            axes[1].plot(
                rates, [per_hold[str(n)]["action_variance_explained"] for n in holds], marker="o", label=name
            )
    axes[0].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[0].set(xscale="log", xlabel="intent update rate (Hz)", ylabel="hold error / total variance",
                title="Staleness of the held intent (1 = no better than the mean)")
    axes[1].axhline(0.0, color="grey", linestyle=":", linewidth=1)
    axes[1].set(xscale="log", xlabel="intent update rate (Hz)", ylabel="share of expert action variance",
                title="Expert action variance reconstructed from the held intent")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--pca_basis", required=True)
    parser.add_argument("--obj_lookahead", type=float, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--holds", default="1,2,3,6,12,30,60,120")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    holds = [int(value) for value in args.holds.split(",")]
    if holds[0] != 1 or sorted(set(holds)) != holds:
        raise ValueError("--holds must be increasing and start at 1")
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
    if not (episode_steps[0] == 0).all():
        raise ValueError("The rollout does not start at episode step 0")

    basis = np.load(args.pca_basis)
    intents = compute_intents(observations, actions, basis, args.obj_lookahead, shape)
    actions = actions.reshape(*shape, ACTION_DIM)
    action_mean = actions.reshape(-1, ACTION_DIM).mean(axis=0)
    action_total_var = float(actions.reshape(-1, ACTION_DIM).var(axis=0).sum())
    goal = intents["goal-4"]

    means = {name: z.reshape(-1, z.shape[-1]).mean(axis=0) for name, z in intents.items()}
    total_var = {name: float(z.reshape(-1, z.shape[-1]).var(axis=0).sum()) for name, z in intents.items()}

    results: dict[str, dict] = {name: {} for name in intents}
    goal_stats: dict[str, dict] = {}
    failures: list[str] = []
    for hold in holds:
        index, valid = held_index(episode_steps, episode_ids, hold)
        count = int(valid.sum())
        age = (np.arange(vector_steps)[:, None] - index)[valid]
        if count != valid.size or age.max() > hold - 1:
            failures.append(f"hold_{hold}_index")
        goal_stats[str(hold)] = {
            "update_rate_hz": CONTROL_HZ / hold,
            "valid_samples": count,
            "mean_intent_age_steps": float(age.mean()),
            "fraction_goal_changed_since_refresh": float(
                (quat_angle(goal, take(goal, index))[valid] > GOAL_CHANGE_ANGLE).mean()
            ),
        }
        for name, z in intents.items():
            held = take(z, index)
            entry = {
                "normalized_hold_error": float(((z - held) ** 2).sum(axis=-1)[valid].mean() / total_var[name]),
            }
            if name.startswith("hand-") or name == "wrist-2":
                error = ((actions - reconstruct_action(name, held, basis, action_mean)) ** 2).sum(axis=-1)
                entry["action_variance_explained"] = float(1.0 - error[valid].mean() / action_total_var)
            if name in QUAT_INTENTS:
                angle = quat_angle(z, held)[valid]
                entry["mean_angle_error_rad"] = float(angle.mean())
                entry["fraction_angle_error_above_success_tolerance"] = float((angle > SUCCESS_TOLERANCE).mean())
            if name == "obj-1":
                entry["mean_abs_error_rad"] = float(np.abs(z - held)[..., 0][valid].mean())
            # Autocorrelation at lag = hold, within episodes.
            if hold < vector_steps:
                same = episode_ids[:-hold] == episode_ids[hold:]
                a, b = z[:-hold] - means[name], z[hold:] - means[name]
                entry["autocorrelation_at_lag"] = float((a * b).sum(axis=-1)[same].mean() / total_var[name])
            results[name][str(hold)] = entry

    # Hold 1 must lose nothing and reproduce the PCA cumulative variance.
    cumulative = np.cumsum(basis["eigenvalues_robot"]) / basis["eigenvalues_robot"].sum()
    for name in intents:
        first = results[name]["1"]
        if first["normalized_hold_error"] != 0.0:
            failures.append(f"{name}_hold1_error")
    if goal_stats["1"]["fraction_goal_changed_since_refresh"] != 0.0:
        failures.append("hold1_goal_changed")
        if name.startswith("hand-"):
            k = int(name.split("-")[1])
            if abs(first["action_variance_explained"] - cumulative[k - 1]) > 1e-3:
                failures.append(f"{name}_hold1_variance")

    changed = (quat_angle(goal[1:], goal[:-1]) > GOAL_CHANGE_ANGLE) & (episode_ids[1:] == episode_ids[:-1])
    pairs = int((episode_ids[1:] == episode_ids[:-1]).sum())
    goal_change_rate = float(changed.sum() / pairs)

    summary = {
        "schema_version": 1,
        "record_type": "shadow_intent_hold_analysis",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "task": manifest.get("task"),
        "pca_basis": str(Path(args.pca_basis).resolve()),
        "pca_basis_sha256": sha256(Path(args.pca_basis)),
        "obj_lookahead_rad": args.obj_lookahead,
        "control_hz": CONTROL_HZ,
        "holds": holds,
        "hold_rule": "refresh when episode_step % N == 0, otherwise keep the previous value",
        "num_samples": int(episode_ids.size),
        "action_total_variance": action_total_var,
        "intent_total_variance": total_var,
        "goal_change_rate_per_step": goal_change_rate,
        "mean_steps_between_goal_changes": (1.0 / goal_change_rate) if goal_change_rate > 0 else None,
        "by_hold": goal_stats,
        "intents": results,
        "failures": failures,
    }

    partial_dir.mkdir(parents=True)
    write_json(partial_dir / "summary.json", summary)
    lines = ["intent,hold_steps,update_rate_hz,normalized_hold_error,autocorrelation_at_lag,"
             "action_variance_explained,mean_angle_error_rad,fraction_angle_error_above_success_tolerance"]
    for name, per_hold in results.items():
        for hold in holds:
            entry = per_hold[str(hold)]
            lines.append(",".join([
                name, str(hold), f"{CONTROL_HZ / hold:g}", f"{entry['normalized_hold_error']:.6f}",
                *(f"{entry[key]:.6f}" if key in entry else "" for key in (
                    "autocorrelation_at_lag", "action_variance_explained", "mean_angle_error_rad",
                    "fraction_angle_error_above_success_tolerance")),
            ]))
    (partial_dir / "intent_hold.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.skip_plot:
        plot(partial_dir / "intent_hold.png", holds, results,
             f"Zero-order hold of intent signals, {episode_ids.size:,} expert transitions ({dataset_dir.name})")
    checksum_lines = [
        f"{sha256(path)}  {path.name}" for path in sorted(partial_dir.iterdir()) if path.name != "SHA256SUMS"
    ]
    (partial_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.replace(partial_dir, output_dir)

    print(f"hold_output_dir={output_dir}")
    print(f"hold_samples={episode_ids.size}")
    print(f"hold_mean_steps_between_goal_changes={summary['mean_steps_between_goal_changes']}")
    for hold in holds:
        stats = goal_stats[str(hold)]
        print(f"hold_goal N={hold} rate_hz={stats['update_rate_hz']:g} "
              f"goal_changed_since_refresh={stats['fraction_goal_changed_since_refresh']:.4f}")
    for name, per_hold in results.items():
        for hold in holds:
            entry = per_hold[str(hold)]
            extra = "".join(
                f" {label}={entry[key]:.4f}" for key, label in (
                    ("autocorrelation_at_lag", "autocorr"), ("action_variance_explained", "action_r2"),
                    ("mean_angle_error_rad", "angle_rad"),
                    ("fraction_angle_error_above_success_tolerance", "above_tol"),
                    ("mean_abs_error_rad", "abs_rad")) if key in entry
            )
            print(f"hold_intent {name} N={hold} error={entry['normalized_hold_error']:.4f}{extra}")
    print(f"hold_failures={failures}")
    print(f"hold_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
