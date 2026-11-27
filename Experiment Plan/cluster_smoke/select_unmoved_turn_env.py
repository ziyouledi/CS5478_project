#!/usr/bin/env python3
"""Pick the video env for the unified task's turn that is never attempted (hinge goal +90 degrees).

Fixed rule: the lowest-index env in which, over the recorded steps, the hinge goal is +90 degrees for more
than 120 steps and the hinge never leaves 0 by more than 0.1 rad (the success tolerance). Reads hinge_angle
and hinge_goal [steps, envs] from the trace written by record_videos.py.

Usage: select_unmoved_turn_env.py TRACE.npz
Prints "selected_env=<k> ..." or "selected_env=none ..." (exit status 1).
"""
import math
import sys

import numpy as np

STALL_STEPS = 120
TOLERANCE = 0.1


def main(path: str) -> int:
    t = np.load(path)
    angle, goal = t["hinge_angle"], t["hinge_goal"]
    plus_steps = (np.abs(goal - math.pi / 2.0) < 0.01).sum(axis=0)
    minus_steps = (np.abs(goal + math.pi / 2.0) < 0.01).sum(axis=0)
    max_angle = np.abs(angle).max(axis=0)
    ok = (plus_steps > STALL_STEPS) & (max_angle <= TOLERANCE)
    summary = (
        f"plus90_steps={' '.join(str(int(v)) for v in plus_steps)} "
        f"minus90_steps={' '.join(str(int(v)) for v in minus_steps)} "
        f"max_abs_hinge_deg={' '.join(f'{math.degrees(v):.1f}' for v in max_angle)}"
    )
    if not ok.any():
        print(f"selected_env=none {summary}")
        return 1
    print(f"selected_env={int(np.argmax(ok))} candidates={int(ok.sum())} of {len(ok)} {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
