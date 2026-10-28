"""Run an Isaac Lab script (e.g. rl_games/train.py) with the M2 task registered.

Usage: ./isaaclab.sh -p run_with_m2_tasks.py scripts/reinforcement_learning/rl_games/train.py [args...]
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_tasks  # noqa: E402,F401

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
