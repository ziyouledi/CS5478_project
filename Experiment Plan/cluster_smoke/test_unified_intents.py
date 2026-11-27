#!/usr/bin/env python3
"""CPU tests of the unified-task (E3) intent code with a synthetic 162-D teacher and rollout. No Isaac Sim.

1. unified_actor_obs layout, and the wrapper's observation-space index list (same expression) in the same order.
2. goal-7 rebuilds the hinge goal and is_turn; hand-K / goal-7 / blind shapes; non-unified intents rejected.
3. validate_intent_modules.py --unified on a fake rollout (teacher = random MLP checkpoint in RL-Games layout,
   PCA basis fitted on its actions) must PASS.

Run: python3 test_unified_intents.py  (prints unified_intent_tests=PASS)
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from intent_modules import (
    FrozenExpert, GOAL_FIELDS, UNIFIED_ACTOR_DIM, UNIFIED_EXT_DIM, UNIFIED_HINGE_ERROR, UNIFIED_HINGE_VEL, UNIFIED_OBS_DIM,
    build_intent, canonical_quat, sha256, unified_actor_obs,
)

HERE = Path(__file__).resolve().parent
N = 4096
TURN = 3


def fake_teacher(path: Path, gen: torch.Generator) -> None:
    sizes = [UNIFIED_OBS_DIM, 64, 32]
    model = {
        "running_mean_std.running_mean": torch.randn(UNIFIED_OBS_DIM, generator=gen) * 0.1,
        "running_mean_std.running_var": torch.rand(UNIFIED_OBS_DIM, generator=gen) + 0.5,
    }
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        model[f"a2c_network.actor_mlp.{2 * i}.weight"] = torch.randn(b, a, generator=gen) / math.sqrt(a)
        model[f"a2c_network.actor_mlp.{2 * i}.bias"] = torch.zeros(b)
    model["a2c_network.mu.weight"] = torch.randn(20, sizes[-1], generator=gen) / math.sqrt(sizes[-1])
    model["a2c_network.mu.bias"] = torch.zeros(20)
    torch.save({"model": model}, path)


def fake_rollout(gen: torch.Generator):
    obs = torch.randn(N, UNIFIED_OBS_DIM, generator=gen)
    q = torch.randn(N, 4, generator=gen)
    obs[:, 64:68] = q / q.norm(dim=1, keepdim=True)
    goal = (torch.randint(-1, 3, (N,), generator=gen).float()) * (math.pi / 2)
    goal = torch.atan2(torch.sin(goal), torch.cos(goal))
    angle = goal + (torch.rand(N, generator=gen) - 0.5) * 2.5
    error = torch.atan2(torch.sin(angle - goal), torch.cos(angle - goal))
    obs[:, 157], obs[:, 158] = torch.sin(error), torch.cos(error)
    kind = torch.randint(0, 4, (N,), generator=gen)
    kind[kind == 2] = 1  # the unified env has no up2
    obs[:, 159] = (kind == TURN).float()
    obs[:, 160] = torch.rand(N, generator=gen)
    return obs, angle, goal, kind


def main() -> None:
    checks: dict[str, bool] = {}
    gen = torch.Generator().manual_seed(0)
    torch.set_grad_enabled(False)

    # 1. layout
    ext = torch.arange(UNIFIED_EXT_DIM, dtype=torch.float32).repeat(2, 1)
    actor = unified_actor_obs(ext)[0].long().tolist()
    expected = list(range(0, 64)) + list(range(72, 157)) + [161, 162, 163]
    checks["actor_layout"] = actor == expected and len(actor) == UNIFIED_ACTOR_DIM == 152
    keep = np.r_[0:GOAL_FIELDS.start, GOAL_FIELDS.stop:UNIFIED_HINGE_ERROR.start, UNIFIED_HINGE_VEL.start]
    checks["wrapper_space_order_matches_actor"] = keep.tolist() + [162, 163] == expected

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "teacher.pth"
        fake_teacher(ckpt, gen)
        obs, angle, goal, kind = fake_rollout(gen)

        expert = FrozenExpert(ckpt, obs_dim=UNIFIED_OBS_DIM)
        actions = expert(obs)
        mu = actions.mean(0)
        cov = torch.cov((actions - mu).T)
        eig, vec = torch.linalg.eigh(cov.double())
        order = torch.argsort(eig, descending=True)
        eig, vec = eig[order].float(), vec[:, order].float()
        basis = tmp / "pca_basis.npz"
        np.savez(basis, mu_robot=mu.numpy(), W_robot=vec.numpy(), eigenvalues_robot=(eig * (N - 1) / N).numpy(),
                 mu_finger=mu[2:].numpy(), W_finger=np.eye(18, dtype=np.float32))
        common = dict(expert_checkpoint=ckpt, expert_sha256=sha256(ckpt), pca_basis=basis, obj_lookahead=0.5)

        # 2. modules
        ext = torch.cat([obs, torch.sin(angle)[:, None], torch.cos(angle)[:, None]], dim=1)
        g7 = build_intent("goal-7", **common, unified=True)(ext)
        checks["goal7_shape"] = tuple(g7.shape) == (N, 7)
        checks["goal7_quat"] = torch.equal(g7[:, :4], canonical_quat(obs[:, 64:68]))
        checks["goal7_hinge_goal"] = float((g7[:, 4:6] - torch.stack([goal.sin(), goal.cos()], 1)).abs().max()) < 1e-5
        checks["goal7_is_turn"] = torch.equal(g7[:, 6], (kind == TURN).float())
        g11 = build_intent("goal-11", **common, unified=True)(ext)
        checks["goal11_shape"] = tuple(g11.shape) == (N, 11)
        checks["goal11_prefix_is_goal7"] = torch.equal(g11[:, :7], g7)
        checks["goal11_relative_rot"] = torch.equal(g11[:, 7:], obs[:, 68:72])
        h8 = build_intent("hand-8", **common, unified=True)
        checks["hand8_uses_first_162"] = torch.allclose(h8(ext), (actions - mu) @ vec[:, :8], atol=1e-5)
        checks["blind_dim0"] = build_intent("blind", **common, unified=True)(ext).shape == (N, 0)
        rejected = []
        for name, kwargs in (("goal-4", {"unified": True}), ("goal-7", {"unified": False}), ("goal-11", {"unified": False}), ("hand-8", {"unified": False})):
            try:
                build_intent(name, **common, **kwargs)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
        checks["wrong_task_rejected"] = all(rejected)  # incl. the 162-D teacher on the 157-D path
        try:
            h8(obs)
            checks["needs_extended_obs"] = False
        except ValueError:
            checks["needs_extended_obs"] = True

        # 3. validation script end to end on a fake rollout directory
        data = tmp / "rollout"
        data.mkdir()
        np.save(data / "observations.npy", obs.numpy())
        np.save(data / "actions.npy", actions.numpy())
        np.save(data / "hinge_angle.npy", angle.numpy())
        np.save(data / "hinge_goal.npy", goal.numpy())
        np.save(data / "phase_kind.npy", kind.numpy().astype(np.uint8))
        out = subprocess.run(
            [sys.executable, str(HERE / "validate_intent_modules.py"), "--unified", "--dataset_dir", str(data),
             "--pca_basis", str(basis), "--expert_checkpoint", str(ckpt), "--expert_sha256", sha256(ckpt),
             "--obj_lookahead", "0.5", "--num_rows", "2048", "--output_json", str(tmp / "report.json")],
            capture_output=True, text=True, cwd=HERE,
        )
        failed = [line for line in out.stdout.splitlines() if line.endswith("FAIL")]
        checks["validate_unified_script_pass"] = out.returncode == 0 and "intent_validation_status=PASS" in out.stdout
        if not checks["validate_unified_script_pass"]:
            print(out.stdout[-3000:], out.stderr[-3000:], failed, sep="\n")

    for name, ok in checks.items():
        print(f"{name}={'PASS' if ok else 'FAIL'}")
    status = all(checks.values())
    print(f"unified_intent_tests={'PASS' if status else 'FAIL'} ({sum(checks.values())}/{len(checks)})")
    raise SystemExit(0 if status else 1)


if __name__ == "__main__":
    main()
