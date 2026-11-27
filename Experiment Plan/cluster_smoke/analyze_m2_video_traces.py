#!/usr/bin/env python3
"""M2 behaviour check: does a policy turn the cube continuously about z (and pass the goals) or stop at each goal?

Input: the <name>_trace.npz files written by record_videos.py (all envs, one value per control step, after the
step). Per env, only steps inside one episode are used (episode_step increases by 1 between them).

Per run (mean over envs):
* yaw_rate_mean_abs: mean |d yaw / dt| (rad/s), yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2)) of the cube.
* yaw_sign_consistency: |sum of signed yaw increments| / sum of |increments| (1 = always the same direction,
  0 = back and forth).
* turns_per_episode: |net yaw| / 2 pi, scaled to a 600-step episode.
* still_fraction: share of steps with |yaw rate| < 0.5 rad/s (the cube is held, not turned).
* tilt_mean_deg: angle between the cube's z axis and world z.
* successes_per_episode: success-counter increments, scaled to 600 steps.
Reading: a continuously turning policy has consistency near 1, few still steps and turns ≈ successes / k for
uniform random yaw goals (k goals passed per turn); a goal-seeking policy stops after each goal (more still steps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STILL_RAD_S = 0.5


def yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(q: np.ndarray) -> np.ndarray:
    x, y = q[..., 1], q[..., 2]
    # z-axis of the rotated frame: its world-z component is 1 - 2(x^2 + y^2)
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))


def analyse(path: Path) -> dict:
    data = np.load(path)
    obj, succ, ep = data["object_rot"], data["successes"], data["episode_step"]
    hz = float(data["control_hz"])
    psi = yaw(obj)
    same_episode = np.diff(ep, axis=0) == 1  # [steps-1, envs]
    dpsi = (np.diff(psi, axis=0) + np.pi) % (2 * np.pi) - np.pi
    dsucc = np.clip(np.diff(succ, axis=0), 0, None)
    per_env = []
    for e in range(obj.shape[1]):
        m = same_episode[:, e]
        if m.sum() < 60:
            continue
        d = dpsi[m, e]
        n = int(m.sum())
        per_env.append({
            "steps": n,
            "yaw_rate_mean_abs": float(np.abs(d).mean() * hz),
            "yaw_sign_consistency": float(abs(d.sum()) / max(np.abs(d).sum(), 1e-9)),
            "turns_per_episode": float(abs(d.sum()) / (2 * np.pi) * 600 / n),
            "still_fraction": float((np.abs(d) * hz < STILL_RAD_S).mean()),
            "tilt_mean_deg": float(tilt_deg(obj[1:][m, e]).mean()),
            "successes_per_episode": float(dsucc[m, e].sum() * 600 / n),
        })
    keys = [k for k in per_env[0] if k != "steps"] if per_env else []
    summary = {k: float(np.mean([p[k] for p in per_env])) for k in keys}
    summary.update({"envs_used": len(per_env), "trace": path.name})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [analyse(p) for p in args.traces]
    for r in rows:
        print(f"{r['trace']:55s} envs={r['envs_used']:2d} |yaw rate|={r.get('yaw_rate_mean_abs', float('nan')):5.2f} rad/s "
              f"consistency={r.get('yaw_sign_consistency', float('nan')):.2f} turns/ep={r.get('turns_per_episode', float('nan')):5.1f} "
              f"still={r.get('still_fraction', float('nan')):.2f} tilt={r.get('tilt_mean_deg', float('nan')):4.1f} deg "
              f"succ/ep={r.get('successes_per_episode', float('nan')):5.1f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"still_threshold_rad_s": STILL_RAD_S, "runs": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
