#!/usr/bin/env python3
"""Priority 2.1 (A0 gate): external-goal wrapper, long-hold stability, and 96-state BFS Oracle on frozen Unified v4.

Tests three things on the frozen Unified v4 skill controller (no retraining):
  1. **Hold stability** (`--hold_steps 60,120,240`): after reaching a goal (`up` or `turn`), keep sending the SAME
     target for H control steps without switching. Measures fall rate, tilt, and hinge drift — this is what v4
     must survive while a VLM is thinking in real time (Priority 5 / A4).
  2. **Closed-loop vs Open-loop BFS Oracle** across stratified optimal lengths L in {1, 2, 3, 4}:
     - Each env starts from a settled in-hand state, gets a random discrete target state (orient_id, detent) at
       exact BFS distance L from its current discrete state.
     - Closed-loop BFS: after each macro-action (reorient or turn, capped at --max_round_steps=120), reads the
       true simulator discrete state and replans via BFS (recovering from v4's ~4.4% hinge drift during reorient).
     - Open-loop BFS: executes the initial BFS plan blindly without replanning.
     - Reports episode success rate per L, rounds used / optimal L, falls, per-action step counts, and hinge drift.

Usage (via 39_p2_bfs_oracle.sbatch on a T4 GPU):
  python p2_bfs_oracle.py --checkpoint ~/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth --output_dir ...
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Unified-HingedCube-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True, help="Frozen Unified v4 checkpoint .pth")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=20261120)
parser.add_argument("--max_round_steps", type=int, default=120, help="Max 60 Hz steps per macro-action")
parser.add_argument("--settle_steps", type=int, default=60, help="Initial steps to settle into a clean discrete state")
parser.add_argument("--hold_durations", default="60,120,240", help="Comma-separated hold steps to test")
parser.add_argument("--levels", default="1,2,3,4", help="BFS optimal step lengths L to evaluate")
parser.add_argument("--episodes_per_level", type=int, default=256, help="Episodes per L for each BFS mode")
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hinged_cube_tasks  # noqa: F401
from cube_state_graph import (
    DIST_TABLE,
    ORIENT_QUATS,
    ORIENTATIONS,
    bfs_plan,
    detent_to_rad,
    index_to_state,
    nearest_orientation,
    rad_to_detent,
    state_to_index,
)
from hinged_cube_env import PHASE_KINDS, QUARTER_TURN, TURN_KIND, _wrap
from intent_modules import sha256

UP_KIND = PHASE_KINDS.index("up")
ANY_KIND = PHASE_KINDS.index("any")


class ExternalGoalUnifiedRunner:
    """Thin external-goal wrapper around ShadowHandUnifiedEnv + frozen Unified v4 policy.

    Overrides the environment's internal phase scheduler so external macro-actions (`reorient` / `turn`)
    directly set (goal_rot, hinge_goal, kind) and run at 60 Hz until completion, fall, or timeout.
    """

    def __init__(self, env, base_env, player, clip_obs: float = math.inf):
        self.env = env
        self.u = base_env.unwrapped
        self.player = player
        self.clip_obs = float(clip_obs)
        self.device = self.u.device
        self.num_envs = self.u.num_envs
        # Disable internal phase timeout termination and prevent internal phase advancing when we drive goals
        self.u.max_phase_steps = 0
        self.u.timeout_terminates = False
        self.orient_quats_t = torch.tensor(ORIENT_QUATS, dtype=torch.float32, device=self.device)
        self.detent_rads_t = torch.tensor([detent_to_rad(d) for d in range(4)], dtype=torch.float32, device=self.device)

    def read_discrete_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (orient_ids, detents, orient_err_deg, detent_res_deg) for all envs."""
        quats = self.u.object_rot.detach().cpu().numpy()
        hinges = self.u.object.data.joint_pos[:, 0].detach().cpu().numpy()
        n = len(quats)
        oids = np.empty(n, dtype=int)
        dets = np.empty(n, dtype=int)
        o_errs = np.empty(n, dtype=float)
        d_errs = np.empty(n, dtype=float)
        for i in range(n):
            oids[i], o_errs[i] = nearest_orientation(quats[i])
            dets[i], d_errs[i] = rad_to_detent(float(hinges[i]))
        return oids, dets, o_errs, d_errs

    def set_goals(self, env_ids: torch.Tensor, actions: list[tuple[str, int]]) -> None:
        """Write external macro-actions into v4's target buffers for `env_ids`."""
        if len(env_ids) == 0:
            return
        cur_hinge = self.u.object.data.joint_pos[env_ids, 0]
        cur_detent_rad = _wrap(torch.round(cur_hinge / QUARTER_TURN) * QUARTER_TURN)
        for idx, (eid, (kind, val)) in enumerate(zip(env_ids.tolist(), actions)):
            if kind == "reorient":
                o = ORIENTATIONS[int(val)]
                self.u.goal_rot[eid] = self.orient_quats_t[int(val)]
                self.u.hinge_goal[eid] = cur_detent_rad[idx]
                self.u.kind[eid] = UP_KIND if o["is_top_up"] else ANY_KIND
            elif kind == "turn":
                # Keep current orientation target aligned to nearest top-up yaw so turn_orient_tol stays valid
                oid, _ = nearest_orientation(self.u.object_rot[eid].detach().cpu().numpy())
                self.u.goal_rot[eid] = self.orient_quats_t[oid]
                target_h = _wrap(cur_detent_rad[idx] + float(val) * QUARTER_TURN)
                self.u.hinge_goal[eid] = target_h
                self.u.kind[eid] = TURN_KIND
            else:
                raise ValueError(f"Unknown action kind: {kind}")
        self.u.phase_steps[env_ids] = 0
        self.u.hold_count[env_ids] = 0
        self.u.max_hold_count[env_ids] = 0
        self.u.phase_start_hinge[env_ids] = cur_hinge
        self.u.reset_goal_buf[env_ids] = 0
        self.u._sync_prev_dists(env_ids)

    def current_obs(self) -> torch.Tensor:
        """Recompute observations directly from current env buffers (reflecting newly written goals)."""
        self.u._compute_intermediate_values()
        obs = self.u.compute_full_observations()
        return torch.clamp(obs, -self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def step_policy(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one 60 Hz step with frozen v4 policy. Returns (next_obs, fallen_mask).

        Note: ShadowHandUnifiedEnv._get_rewards() auto-advances goals on `done`. To keep external control,
        we snapshot target buffers before step() and restore them if the env tried to auto-advance.
        """
        saved_goal_rot = self.u.goal_rot.clone()
        saved_hinge_goal = self.u.hinge_goal.clone()
        saved_kind = self.u.kind.clone()

        actions = self.player.get_action(self.player.obs_to_torch(obs), is_deterministic=True)
        _, _, _, _ = self.env.step(actions)

        goal_dist = torch.norm(self.u.object_pos - self.u.in_hand_pos, p=2, dim=-1)
        fallen = (goal_dist >= self.u.cfg.fall_dist) | self.u.reset_terminated.bool()

        # Keep external goals fixed (do not let internal scheduler overwrite them)
        self.u.goal_rot[:] = saved_goal_rot
        self.u.hinge_goal[:] = saved_hinge_goal
        self.u.kind[:] = saved_kind
        self.u.reset_goal_buf[:] = 0
        next_obs = self.current_obs()
        return next_obs, fallen

    @torch.no_grad()
    def run_hold_stability(self, durations: list[int], max_reach_steps: int = 120) -> dict:
        """Test keeping the same goal after completion for H steps (A0 check 3)."""
        results = {}
        all_ids = torch.arange(self.num_envs, device=self.device)
        for test_kind in ("up", "turn"):
            self.env.reset()
            # First reach a clean `up` orientation
            self.set_goals(all_ids, [("reorient", 0)] * self.num_envs)
            obs = self.current_obs()
            for _ in range(80):
                obs, _ = self.step_policy(obs)
            if test_kind == "turn":
                self.set_goals(all_ids, [("turn", +1 if i % 2 == 0 else -1) for i in range(self.num_envs)])
            else:
                self.set_goals(all_ids, [("reorient", 1)] * self.num_envs)
            obs = self.current_obs()

            # Wait until envs reach the goal (up to max_reach_steps)
            reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            hold_cnt = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for _ in range(max_reach_steps):
                obs, fallen = self.step_policy(obs)
                orient_err = torch.abs(self.u._hinge_error()) if False else None
                from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import rotation_distance
                od = torch.abs(rotation_distance(self.u.object_rot, self.u.goal_rot))
                hd = torch.abs(self.u._hinge_error())
                if test_kind == "turn":
                    within = (od <= self.u.turn_orient_tol) & (hd <= self.u.cfg.success_tolerance)
                    hold_cnt = torch.where(within, hold_cnt + 1, torch.zeros_like(hold_cnt))
                    reached |= (hold_cnt >= self.u.hold_steps) & (~fallen)
                else:
                    reached |= (od <= self.u.cfg.success_tolerance) & (hd <= self.u.cfg.success_tolerance) & (~fallen)
                if reached.all():
                    break

            # Now hold the SAME goal for max(durations) steps and record metrics at each duration checkpoint
            max_h = max(durations)
            ever_fell = ~reached.clone()
            n_reached = int(reached.sum().item())
            start_hinge = self.u.object.data.joint_pos[:, 0].clone()
            per_h = {}
            for step in range(1, max_h + 1):
                obs, fallen = self.step_policy(obs)
                ever_fell |= fallen
                if step in durations:
                    alive = reached & (~ever_fell)
                    tilt = self.u._tilt_deg()
                    h_drift = torch.rad2deg(torch.abs(_wrap(self.u.object.data.joint_pos[:, 0] - start_hinge)))
                    per_h[str(step)] = {
                        "n_started_reached": n_reached,
                        "fall_rate": float((reached & ever_fell).sum().item()) / max(n_reached, 1),
                        "mean_tilt_deg": float(tilt[alive].mean().item()) if alive.any() else None,
                        "mean_hinge_drift_deg": float(h_drift[alive].mean().item()) if alive.any() else None,
                        "frac_hinge_drift_over_45": float((h_drift[alive] > 45.0).float().mean().item()) if alive.any() else None,
                    }
            results[test_kind] = per_h
        return results

    @torch.no_grad()
    def run_bfs_oracle(self, levels: list[int], mode: str, max_round_steps: int, rng: np.random.Generator) -> dict:
        """Evaluate Closed-loop or Open-loop BFS Oracle on stratified optimal step lengths L."""
        from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import rotation_distance

        assert mode in ("closed_loop", "open_loop")
        all_ids = torch.arange(self.num_envs, device=self.device)
        level_results = {}

        for L in levels:
            # Reset and settle into clean starting states that actually have at least one target at distance L
            # (Note: L=4 requires a non-top-up starting orientation, since top-up states have max eccentricity 3)
            self.env.reset()
            valid_start_oids = [o for o in range(24) if any((DIST_TABLE[state_to_index(o, d)] == L).any() for d in range(4))]
            init_oids = rng.choice(valid_start_oids, size=self.num_envs)
            self.set_goals(all_ids, [("reorient", int(o)) for o in init_oids])
            obs = self.current_obs()
            for _ in range(args_cli.settle_steps):
                obs, _ = self.step_policy(obs)

            src_oids, src_dets, _, _ = self.read_discrete_states()
            dst_oids = np.empty(self.num_envs, dtype=int)
            dst_dets = np.empty(self.num_envs, dtype=int)
            open_plans: list[list[tuple[str, int]]] = []
            for i in range(self.num_envs):
                s_idx = state_to_index(int(src_oids[i]), int(src_dets[i]))
                cands = np.nonzero(DIST_TABLE[s_idx] == L)[0]
                if len(cands) == 0:
                    # Fallback if an env failed to settle to non-top-up: use max available distance from s_idx
                    cands = np.nonzero(DIST_TABLE[s_idx] == DIST_TABLE[s_idx].max())[0]
                t_idx = int(rng.choice(cands))
                to, td = index_to_state(t_idx)
                dst_oids[i], dst_dets[i] = to, td
                open_plans.append(bfs_plan((int(src_oids[i]), int(src_dets[i])), (to, td)))

            max_rounds = 2 * L + 2 if mode == "closed_loop" else L
            active = np.ones(self.num_envs, dtype=bool)
            succeeded = np.zeros(self.num_envs, dtype=bool)
            fell_ep = np.zeros(self.num_envs, dtype=bool)
            rounds_used = np.zeros(self.num_envs, dtype=int)
            total_steps = np.zeros(self.num_envs, dtype=int)
            replan_events = 0
            action_stats = collections.defaultdict(lambda: {"started": 0, "reached": 0, "steps": [], "hinge_drift_over_45": 0})

            for rnd in range(max_rounds):
                cur_oids, cur_dets, _, _ = self.read_discrete_states()
                # Check if any active env is already at dst
                for i in range(self.num_envs):
                    if active[i] and cur_oids[i] == dst_oids[i] and cur_dets[i] == dst_dets[i]:
                        succeeded[i] = True
                        active[i] = False
                if not active.any():
                    break

                active_indices = np.nonzero(active)[0]
                round_actions = []
                valid_eids = []
                for i in active_indices:
                    if mode == "closed_loop":
                        plan = bfs_plan((int(cur_oids[i]), int(cur_dets[i])), (int(dst_oids[i]), int(dst_dets[i])))
                        if rnd > 0 and len(plan) >= len(open_plans[i]):
                            replan_events += 1
                        act = plan[0]
                    else:
                        if rnd >= len(open_plans[i]):
                            active[i] = False
                            continue
                        act = open_plans[i][rnd]
                    valid_eids.append(int(i))
                    round_actions.append(act)
                    rounds_used[i] += 1
                    action_stats[act[0]]["started"] += 1

                if not valid_eids:
                    break

                eid_t = torch.tensor(valid_eids, dtype=torch.long, device=self.device)
                start_hinge = self.u.object.data.joint_pos[eid_t, 0].clone()
                self.set_goals(eid_t, round_actions)
                obs = self.current_obs()

                done_round = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                round_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                round_steps_arr = np.full(self.num_envs, max_round_steps, dtype=int)
                is_turn_mask = self.u.kind == TURN_KIND

                for st in range(1, max_round_steps + 1):
                    obs, fallen = self.step_policy(obs)
                    od = torch.abs(rotation_distance(self.u.object_rot, self.u.goal_rot))
                    hd = torch.abs(self.u._hinge_error())
                    orient_tol = torch.where(is_turn_mask, self.u.turn_orient_tol, self.u.cfg.success_tolerance)
                    within = (od <= orient_tol) & (hd <= self.u.cfg.success_tolerance)
                    round_hold = torch.where(within, round_hold + 1, torch.zeros_like(round_hold))
                    req_hold = torch.where(is_turn_mask, self.u.hold_steps, 1)
                    just_done = (~done_round) & (round_hold >= req_hold) & (~fallen)
                    for eid in valid_eids:
                        if not active[eid]:
                            continue
                        if fallen[eid].item():
                            fell_ep[eid] = True
                            active[eid] = False
                            done_round[eid] = True
                            round_steps_arr[eid] = st
                        elif just_done[eid].item():
                            done_round[eid] = True
                            round_steps_arr[eid] = st
                    if all(done_round[eid].item() or not active[eid] for eid in valid_eids):
                        break

                end_hinge = self.u.object.data.joint_pos[eid_t, 0]
                drift_deg = torch.rad2deg(torch.abs(_wrap(end_hinge - start_hinge))).cpu().numpy()
                for idx, (eid, act) in enumerate(zip(valid_eids, round_actions)):
                    total_steps[eid] += round_steps_arr[eid]
                    if done_round[eid].item() and not fell_ep[eid]:
                        action_stats[act[0]]["reached"] += 1
                        action_stats[act[0]]["steps"].append(int(round_steps_arr[eid]))
                        if act[0] == "reorient" and drift_deg[idx] > 45.0:
                            action_stats[act[0]]["hinge_drift_over_45"] += 1

            # Final state check after last round
            fin_oids, fin_dets, _, _ = self.read_discrete_states()
            for i in range(self.num_envs):
                if active[i] and not fell_ep[i] and fin_oids[i] == dst_oids[i] and fin_dets[i] == dst_dets[i]:
                    succeeded[i] = True

            act_summary = {}
            for k, v in action_stats.items():
                st_arr = np.asarray(v["steps"], dtype=float)
                act_summary[k] = {
                    "started": v["started"],
                    "reached": v["reached"],
                    "reach_rate": v["reached"] / max(v["started"], 1),
                    "steps_median": float(np.median(st_arr)) if len(st_arr) else None,
                    "steps_mean": float(np.mean(st_arr)) if len(st_arr) else None,
                    "hinge_drift_over_45_rate": v["hinge_drift_over_45"] / max(v["reached"], 1),
                }

            level_results[str(L)] = {
                "n_episodes": self.num_envs,
                "success_rate": float(succeeded.mean()),
                "fall_rate": float(fell_ep.mean()),
                "mean_rounds_succeeded": float(rounds_used[succeeded].mean()) if succeeded.any() else None,
                "rounds_ratio_vs_optimal": float(rounds_used[succeeded].mean() / L) if succeeded.any() else None,
                "mean_control_steps_succeeded": float(total_steps[succeeded].mean()) if succeeded.any() else None,
                "replan_events": replan_events,
                "actions": act_summary,
            }
            print(
                f"p2_bfs mode={mode} L={L} succ={succeeded.mean():.3f} fall={fell_ep.mean():.3f} "
                f"rounds_ok={level_results[str(L)]['mean_rounds_succeeded']} steps_ok={level_results[str(L)]['mean_control_steps_succeeded']}",
                flush=True,
            )
        return level_results


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    t0 = time.perf_counter()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(base_env, args_cli.device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()
    player.has_batch_dimension = True

    wrapper = ExternalGoalUnifiedRunner(env, base_env, player, clip_obs=clip_obs)
    durations = [int(x) for x in args_cli.hold_durations.split(",") if x.strip()]
    levels = [int(x) for x in args_cli.levels.split(",") if x.strip()]
    rng = np.random.default_rng(args_cli.seed)

    print("Running A0 hold stability test...", flush=True)
    hold_res = wrapper.run_hold_stability(durations, max_reach_steps=args_cli.max_round_steps)
    print("p2_hold=" + json.dumps(hold_res), flush=True)

    print("Running Closed-loop BFS Oracle...", flush=True)
    closed_res = wrapper.run_bfs_oracle(levels, "closed_loop", args_cli.max_round_steps, rng)

    print("Running Open-loop BFS Oracle...", flush=True)
    open_res = wrapper.run_bfs_oracle(levels, "open_loop", args_cli.max_round_steps, rng)

    gate_l2_pass = closed_res.get("2", {}).get("success_rate", 0.0) >= 0.70
    summary = {
        "record_type": "p2_bfs_oracle",
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "max_round_steps": args_cli.max_round_steps,
        "elapsed_s": time.perf_counter() - t0,
        "hold_stability": hold_res,
        "bfs_closed_loop": closed_res,
        "bfs_open_loop": open_res,
        "gate_l2_pass": gate_l2_pass,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    env.close()
    print(f"p2_bfs_oracle_status=PASS gate_l2_pass={gate_l2_pass}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("p2_bfs_oracle_status=FAIL", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
