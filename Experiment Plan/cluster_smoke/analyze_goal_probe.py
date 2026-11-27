#!/usr/bin/env python3
"""Offline probe: how much goal information does a held hand-k action code carry? (CPU, no Isaac Sim)

On a recorded expert rollout, the target is the hidden goal error the goal-hidden student cannot see:
the rotation vector of obs[68:72] = quat_mul(object_rot, conj(goal_rot)) (Stage 1 / M1) or
hinge_rot * conj(goal_rot) (face-turn). Features, all at the same control step t:

* ``state``        the goal-hidden actor state (obs without [64:72]; face-turn adds sin/cos of the hinge angle);
* ``code``         the hand-k code held since the last refresh (refresh when episode_step % N == 0);
* ``state+code``   both.

For each feature set a linear (ridge) and an MLP regressor are fitted on 80% of the environments and scored
on the rest: R^2 = 1 - SSE / total variance, pooled over the 3 components. The goal information the held code
adds beyond the state is R^2(state+code) - R^2(state). Probes are a lower bound on the information present.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analyze_intent_hold import held_index, take
from intent_modules import GOAL_FIELDS, OBS_DIM, hinge_angle_obs, quat_to_rotvec, sha256

ACTION_DIM = 20
CONTROL_HZ = 60.0
GOAL_ERROR = slice(68, 72)
TEST_FRACTION = 0.2


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def r2(pred: np.ndarray, target: np.ndarray, mean: np.ndarray) -> float:
    return float(1.0 - ((pred - target) ** 2).sum() / ((target - mean) ** 2).sum())


def fit_linear(x_tr, y_tr, x_te, ridge: float = 1e-3) -> np.ndarray:
    xa = np.hstack([x_tr, np.ones((len(x_tr), 1))])
    w = np.linalg.solve(xa.T @ xa + ridge * len(xa) * np.eye(xa.shape[1]), xa.T @ y_tr)
    return np.hstack([x_te, np.ones((len(x_te), 1))]) @ w


def fit_mlp(x_tr, y_tr, x_te, steps: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x_tr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, y_tr.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xt, yt = torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).float()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (4096,), generator=gen)
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in torch.from_numpy(x_te).float().split(65536)]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--pca_basis", required=True)
    parser.add_argument("--face_turn", action="store_true")
    parser.add_argument("--k", type=int, default=8, help="hand-k code size")
    parser.add_argument("--holds", default="1,3,6,12,30")
    parser.add_argument("--max_train", type=int, default=400000)
    parser.add_argument("--max_test", type=int, default=200000)
    parser.add_argument("--mlp_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(True)
    holds = [int(v) for v in args.holds.split(",")]
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"Output or partial directory already exists: {output_dir}")

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_envs, steps_total = int(manifest["num_envs"]), int(manifest["vector_steps"])
    shape = (steps_total, num_envs)
    obs = np.asarray(np.load(dataset_dir / "observations.npy", mmap_mode="r"), dtype=np.float32)
    actions = np.asarray(np.load(dataset_dir / "actions.npy", mmap_mode="r"), dtype=np.float64)
    env_ids = np.load(dataset_dir / "env_ids.npy", mmap_mode="r")
    if obs.shape != (num_envs * steps_total, OBS_DIM) or actions.shape != (num_envs * steps_total, ACTION_DIM):
        raise ValueError(f"Unexpected shapes: obs {obs.shape}, actions {actions.shape}")
    if not np.array_equal(env_ids[:num_envs], np.arange(num_envs)):
        raise ValueError("Rows are not ordered vector-step-major with env IDs 0..num_envs-1")
    episode_ids = np.asarray(np.load(dataset_dir / "episode_ids.npy"), dtype=np.int64).reshape(shape)
    episode_steps = np.asarray(np.load(dataset_dir / "episode_steps.npy"), dtype=np.int64).reshape(shape)

    basis = np.load(args.pca_basis)
    code = ((actions - basis["mu_robot"]) @ basis["W_robot"][:, : args.k]).reshape(*shape, args.k)
    obs_t = torch.from_numpy(obs)
    target = quat_to_rotvec(obs_t[:, GOAL_ERROR]).double().numpy().reshape(*shape, 3)
    keep = np.ones(OBS_DIM, dtype=bool)
    keep[GOAL_FIELDS] = False
    state = obs[:, keep]
    if args.face_turn:
        state = np.hstack([state, hinge_angle_obs(obs_t).numpy()])
    state = state.reshape(*shape, -1)
    del obs_t

    rng = np.random.default_rng(args.seed)
    test_envs = np.zeros(num_envs, dtype=bool)
    test_envs[rng.choice(num_envs, int(round(TEST_FRACTION * num_envs)), replace=False)] = True

    def sample(mask: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        t, e = np.nonzero(mask)
        pick = rng.choice(len(t), min(limit, len(t)), replace=False)
        return t[pick], e[pick]

    valid_all = np.ones(shape, dtype=bool)
    tr = sample(valid_all & ~test_envs[None, :], args.max_train)
    te = sample(valid_all & test_envs[None, :], args.max_test)
    y_tr, y_te = target[tr], target[te]
    y_mean = y_tr.mean(axis=0)

    def standardize(a_tr, a_te):
        mu, sd = a_tr.mean(axis=0), a_tr.std(axis=0) + 1e-6
        return (a_tr - mu) / sd, (a_te - mu) / sd

    results: dict[str, dict] = {}

    def probe(name: str, x_tr: np.ndarray, x_te: np.ndarray) -> None:
        x_tr, x_te = standardize(x_tr, x_te)
        entry = {
            "linear_r2": r2(fit_linear(x_tr, y_tr, x_te), y_te, y_mean),
            "mlp_r2": r2(fit_mlp(x_tr, y_tr, x_te, args.mlp_steps, args.seed), y_te, y_mean),
        }
        results[name] = entry
        print(f"probe {name} linear_r2={entry['linear_r2']:.4f} mlp_r2={entry['mlp_r2']:.4f}", flush=True)

    probe("state", state[tr], state[te])
    for hold in holds:
        index, valid = held_index(episode_steps, episode_ids, hold)
        if not valid.all():
            raise ValueError(f"hold {hold}: invalid held index")
        held = take(code, index)
        probe(f"code_N{hold}", held[tr], held[te])
        probe(f"state+code_N{hold}", np.hstack([state[tr], held[tr]]), np.hstack([state[te], held[te]]))

    failures = [name for name, e in results.items() if not all(np.isfinite(v) for v in e.values())]
    summary = {
        "schema_version": 1,
        "record_type": "shadow_goal_probe",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": sha256(manifest_path),
        "task": manifest.get("task"),
        "pca_basis": str(Path(args.pca_basis).resolve()),
        "pca_basis_sha256": sha256(Path(args.pca_basis)),
        "face_turn": args.face_turn,
        "k": args.k,
        "holds": holds,
        "target": "rotation vector of obs[68:72] (hidden goal error)",
        "state_dim": int(state.shape[-1]),
        "train_samples": int(len(y_tr)),
        "test_samples": int(len(y_te)),
        "split": f"by environment, {TEST_FRACTION:.0%} of envs held out",
        "mlp": f"2x256 ReLU, Adam 1e-3 cosine, {args.mlp_steps} steps x 4096",
        "seed": args.seed,
        "results": results,
        "failures": failures,
    }
    partial_dir.mkdir(parents=True)
    write_json(partial_dir / "summary.json", summary)
    lines = ["features,hold_steps,update_rate_hz,linear_r2,mlp_r2"]
    for name, e in results.items():
        hold = int(name.split("_N")[1]) if "_N" in name else ""
        rate = f"{CONTROL_HZ / hold:g}" if hold else ""
        lines.append(f"{name.split('_N')[0]},{hold},{rate},{e['linear_r2']:.6f},{e['mlp_r2']:.6f}")
    (partial_dir / "goal_probe.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_lines = [f"{sha256(p)}  {p.name}" for p in sorted(partial_dir.iterdir()) if p.name != "SHA256SUMS"]
    (partial_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.replace(partial_dir, output_dir)
    print(f"probe_output_dir={output_dir}")
    print(f"probe_failures={failures}")
    print(f"probe_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
