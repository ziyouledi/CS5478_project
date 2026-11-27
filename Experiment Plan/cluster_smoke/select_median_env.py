#!/usr/bin/env python3
"""Pick the demo-video env by a fixed rule: the env whose successes over the recorded steps are closest to the
median over all envs of the same recording (ties -> lowest index). Pre-specified, so no cherry-picking.

Successes over the clip = sum of positive steps of the success counter (it is zeroed on reset), i.e. what a
viewer of the 600-step clip would count, including episodes that restart inside the clip.

Usage: select_median_env.py TRACE.npz [--json]
Prints "selected_env=<k> successes=<s> median=<m> all=<...>".
"""
import json
import sys

import numpy as np


def clip_successes(successes: np.ndarray) -> np.ndarray:
    """successes: [steps, envs] counter after each step -> successes per env over the clip."""
    prev = np.vstack([np.zeros((1, successes.shape[1])), successes[:-1]])
    step = successes - prev
    return np.where(step > 0, step, 0).sum(axis=0)


def select(successes: np.ndarray) -> tuple[int, np.ndarray, float]:
    per_env = clip_successes(successes)
    median = float(np.median(per_env))
    k = int(np.argmin(np.abs(per_env - median)))  # argmin returns the lowest index on ties
    return k, per_env, median


def main(path: str, as_json: bool) -> None:
    t = np.load(path)
    k, per_env, median = select(t["successes"])
    if as_json:
        print(json.dumps({"selected_env": k, "successes": float(per_env[k]), "median": median,
                          "per_env": per_env.tolist()}))
    else:
        print(f"selected_env={k} successes={per_env[k]:g} median={median:g} all={' '.join(f'{v:g}' for v in per_env)}")


if __name__ == "__main__":
    main(sys.argv[1], "--json" in sys.argv[2:])
