"""Pure-torch helpers for student evaluation (no Isaac Sim imports)."""

from __future__ import annotations

import re
from pathlib import Path

import torch


CHECKPOINT_ITERATION = re.compile(r"_ep_(\d+)_")


def checkpoint_iteration(path: Path) -> int | None:
    """RL-Games periodic checkpoints are named ``last_<name>_ep_<iteration>_rew_<reward>.pth``."""
    match = CHECKPOINT_ITERATION.search(Path(path).name)
    return int(match.group(1)) if match else None


def list_checkpoints(nn_dir: Path, stride: int = 1) -> list[tuple[int, Path]]:
    """Periodic checkpoints sorted by iteration; keeps every ``stride``-th one plus the last.

    When several files share an iteration, the most recently written one is used.
    """
    by_iteration: dict[int, Path] = {}
    for path in Path(nn_dir).glob("*.pth"):
        iteration = checkpoint_iteration(path)
        if iteration is None:
            continue
        if iteration not in by_iteration or path.stat().st_mtime >= by_iteration[iteration].stat().st_mtime:
            by_iteration[iteration] = path
    found = sorted(by_iteration.items())
    if stride > 1 and found:
        kept = found[::stride]
        if kept[-1] != found[-1]:
            kept.append(found[-1])
        found = kept
    return found


class FirstEpisodeTracker:
    """Records only the first episode of each environment, so short (failed) episodes are not
    over-represented the way "first N completed episodes" would over-represent them.

    ``record_successes`` must be called with the environment's per-env success counter just
    before ``_reset_idx`` zeroes it; ``step`` is called after each vector step.
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.successes = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
        self.timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active = False

    def record_successes(self, env_ids: torch.Tensor, successes: torch.Tensor) -> None:
        if not self.active:
            return
        env_ids = torch.as_tensor(env_ids, device=self.successes.device).long().flatten()
        values = torch.as_tensor(successes, device=self.successes.device).long().flatten()
        first = ~self.finished[env_ids]
        self.successes[env_ids[first]] = values[first]

    def step(self, rewards: torch.Tensor, dones: torch.Tensor, time_outs: torch.Tensor | None) -> None:
        open_envs = ~self.finished
        self.returns += torch.where(open_envs, rewards.double().flatten(), torch.zeros_like(self.returns))
        self.lengths += open_envs.long()
        newly_done = open_envs & dones.bool().flatten()
        if time_outs is not None:
            self.timed_out |= newly_done & time_outs.bool().flatten()
        self.finished |= newly_done

    @property
    def all_finished(self) -> bool:
        return bool(self.finished.all())

    def summary(self) -> dict:
        if not self.all_finished:
            raise RuntimeError(f"{int((~self.finished).sum())} environments have not finished their first episode")
        if bool((self.successes < 0).any()):
            raise RuntimeError("Some finished episodes have no recorded success count")
        successes = self.successes.double()
        returns = self.returns
        fell = (~self.timed_out).double()
        return {
            "episodes": self.num_envs,
            "mean_successes_per_episode": float(successes.mean()),
            "std_successes_per_episode": float(successes.std(unbiased=False)),
            "median_successes_per_episode": float(successes.median()),
            "fraction_episodes_with_success": float((successes > 0).double().mean()),
            "mean_episode_reward": float(returns.mean()),
            "std_episode_reward": float(returns.std(unbiased=False)),
            "mean_episode_length": float(self.lengths.double().mean()),
            "fall_fraction": float(fell.mean()),
            "per_episode_successes": self.successes.cpu().tolist(),
            "per_episode_reward": returns.cpu().tolist(),
        }
