#!/usr/bin/env python3
"""CPU tests of intent_modules.selector_view (FACE_TURN_PLAN.md E2'). Pure torch, no Isaac Sim.

Run: python3 test_selector_view.py  (prints selector_view_tests=PASS)
"""

from __future__ import annotations

import math

import torch

from intent_modules import IntentChannel, quat_conjugate, quat_mul, selector_view

TURN = 3
N_ENVS = 4


def z_rot(angle: float | torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=torch.float32)
    return torch.stack((torch.cos(angle / 2), torch.zeros_like(angle), torch.zeros_like(angle), torch.sin(angle / 2)), -1)


def env_fields(kind, goal, obj, hinge):
    """The env's own obs[64:72] (ShadowHandSwitchEnv.compute_full_observations)."""
    current = torch.where((kind == TURN).unsqueeze(-1), hinge, obj)
    return torch.cat((goal, quat_mul(current, quat_conjugate(goal))), -1)


def random_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


def run(channel: IntentChannel, script):
    """script: list of (step [N], kind [N], goal [N,4]); obj / hinge vary every step."""
    gen = torch.Generator().manual_seed(0)
    out = []
    for step, kind, goal in script:
        obj, hinge = random_quat(N_ENVS, gen), z_rot(torch.rand(N_ENVS, generator=gen) * 3)
        k, f = selector_view(channel, kind, goal, obj, hinge, step, TURN)
        out.append((k, f, obj, hinge))
    return out


def make_script(steps: int, change_at: int, reset_env: int | None = None, reset_at: int | None = None):
    """Env 0..N-1 in kind 1 ('up') with goal A until change_at, then kind TURN with goal B (a completed phase).
    Optionally one env resets at reset_at (episode step back to 0, kind 1, goal C)."""
    goal_a, goal_b, goal_c = z_rot(0.3), z_rot(math.pi / 2), z_rot(-1.0)
    script = []
    for t in range(steps):
        step = torch.full((N_ENVS,), t, dtype=torch.long)
        kind = torch.full((N_ENVS,), 1 if t < change_at else TURN, dtype=torch.long)
        goal = (goal_a if t < change_at else goal_b).repeat(N_ENVS, 1)
        if reset_env is not None and t >= reset_at:
            step[reset_env] = t - reset_at
            kind[reset_env] = 1
            goal[reset_env] = goal_c
        script.append((step, kind, goal))
    return script


def close(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol)


def main() -> None:
    checks = {}

    # 1. pass-through: message = env, fields = env fields, every step
    script = make_script(12, change_at=5)
    out = run(IntentChannel(), script)
    checks["passthrough_equals_env"] = all(
        torch.equal(k, kind) and torch.equal(f, env_fields(kind, goal, obj, hinge))
        for (step, kind, goal), (k, f, obj, hinge) in zip(script, out)
    )

    # 2. hold 6: change at step 5 (not a multiple of 6) becomes visible at step 6, not before
    out = run(IntentChannel(hold=6), script)
    ok = True
    for t, ((step, kind, goal), (k, f, obj, hinge)) in enumerate(zip(script, out)):
        seen_kind, seen_goal = (1, z_rot(0.3)) if t < 6 else (TURN, z_rot(math.pi / 2))
        expect = env_fields(torch.full_like(kind, seen_kind), seen_goal.repeat(N_ENVS, 1), obj, hinge)
        ok &= bool((k == seen_kind).all()) and close(f, expect)
    checks["hold6_refresh_at_multiple"] = ok
    # stale old goal is measured against the CURRENT rotation in the OLD layout (cube, not hinge)
    k5, f5, obj5, _ = out[5]
    checks["hold6_stale_uses_old_layout"] = bool((k5 == 1).all()) and close(
        f5[:, 4:8], quat_mul(obj5, quat_conjugate(z_rot(0.3).repeat(N_ENVS, 1)))
    )

    # 3. delay 2: step t sees the message of step max(t - 2, 0)
    out = run(IntentChannel(delay=2), script)
    ok = True
    for t, (_, (k, f, obj, hinge)) in enumerate(zip(script, out)):
        src_kind, src_goal = script[max(t - 2, 0)][1], script[max(t - 2, 0)][2]
        ok &= torch.equal(k, src_kind) and close(f, env_fields(src_kind, src_goal, obj, hinge))
    checks["delay2_lags_two_steps"] = ok
    checks["delay2_turn_visible_at_7"] = bool((out[6][0] == 1).all() and (out[7][0] == TURN).all())

    # 4. reset of one env under delay: from its episode step 0 it sees its new episode's message,
    #    never the previous episode's (same rule as the rate sweep intent channel)
    script = make_script(14, change_at=4, reset_env=2, reset_at=9)
    out = run(IntentChannel(delay=3), script)
    ok = True
    for t in range(9, 14):
        k, f, obj, hinge = out[t]
        ok &= int(k[2]) == 1 and close(f[2, 0:4], z_rot(-1.0))
    checks["delay_reset_starts_fresh"] = ok
    # the other envs are unaffected by env 2's reset
    checks["delay_reset_others_unchanged"] = bool((out[13][0][[0, 1, 3]] == TURN).all())

    # 5. hold: reset env refreshes at its episode step 0
    out = run(IntentChannel(hold=30), script)
    checks["hold_reset_refreshes"] = int(out[9][0][2]) == 1 and close(out[9][1][2, 0:4], z_rot(-1.0)) and bool(
        (out[13][0][[0, 1, 3]] == 1).all()  # others still hold the step-0 message (no refresh until 30)
    )

    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    status = all(checks.values())
    print(f"selector_view_tests={'PASS' if status else 'FAIL'} ({sum(checks.values())}/{len(checks)})")
    raise SystemExit(0 if status else 1)


if __name__ == "__main__":
    main()
