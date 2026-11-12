"""Intent interfaces for Shadow Hand student policies (pure torch, no Isaac Sim imports).

Every intent maps the clipped 157-D policy observation of the student's current state to a
k-D vector that is appended to the student's observation:

* ``blind``            0-D, no intent.
* ``hand-K``           first K principal components of the frozen expert's 20-D action.
* ``joint-K``          the expert's 20-D action projected on the first K components of its measured
                       actuated joint positions (basis from fit_joint_basis.py; same normalized units).
* ``wrist-2``          the expert's two wrist targets, uncompressed.
* ``wrist2-finger-K``  wrist targets plus the first K components of the 18-D finger PCA.
* ``obj-1``            remaining rotation angle to the goal.
* ``obj-3``            rotation vector from the current cube orientation to the subgoal.
* ``obj-4``            subgoal quaternion slerp(q_obj, q_goal, min(1, lookahead / theta_remain)).
* ``goal-4``           the goal quaternion itself, w >= 0 (goal-level interface; no expert, no lookahead).
* ``goal-yaw``         (sin psi, cos psi), psi = 2 atan2(q_z, q_w) of the goal quaternion: its rotation angle
                       about world z. Only meaningful on the M2 task, whose goals are pure yaw.

``IntentChannel`` models the link between the intent source and the actor: zero-order hold, delay and
quantisation (rate sweep).

Quaternions follow Isaac Lab's (w, x, y, z) convention.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


OBS_DIM = 157
ACTION_DIM = 20
OBJECT_ROT = slice(51, 55)
GOAL_ROT = slice(64, 68)
WRIST = slice(0, 2)
FINGERS = slice(2, 20)
# Stage 2 M1 hides the goal from the actor: goal_rot [64:68] and quat_mul(object_rot, conj(goal_rot))
# [68:72]. in_hand_pos [61:64] is a fixed target position and stays.
GOAL_FIELDS = slice(64, 72)
GOAL_HIDDEN_OBS_DIM = OBS_DIM - (GOAL_FIELDS.stop - GOAL_FIELDS.start)
EXPERT_OBS_EPSILON = 1e-5
EXPERT_OBS_NORM_CLIP = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Quaternion helpers, (w, x, y, z).


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], dim=1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=1)
    bw, bx, by, bz = b.unbind(dim=1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=1,
    )


def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    """Same formula as Isaac Lab's in-hand manipulation reward."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
    """Shortest-path spherical interpolation; fraction has shape [N]."""
    q0 = q0 / q0.norm(dim=1, keepdim=True)
    q1 = q1 / q1.norm(dim=1, keepdim=True)
    dot = (q0 * q1).sum(dim=1)
    q1 = torch.where((dot < 0)[:, None], -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    small = sin_omega < 1e-6
    safe_sin = torch.where(small, torch.ones_like(sin_omega), sin_omega)
    w0 = torch.where(small, 1.0 - fraction, torch.sin((1.0 - fraction) * omega) / safe_sin)
    w1 = torch.where(small, fraction, torch.sin(fraction * omega) / safe_sin)
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / out.norm(dim=1, keepdim=True)


def canonical_quat(q: torch.Tensor) -> torch.Tensor:
    """Pick the w >= 0 representative so q and -q map to one input."""
    return torch.where((q[:, :1] < 0), -q, q)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = canonical_quat(q / q.norm(dim=1, keepdim=True))
    vec_norm = q[:, 1:].norm(dim=1)
    angle = 2.0 * torch.atan2(vec_norm, q[:, 0])
    scale = torch.where(vec_norm > 1e-8, angle / vec_norm.clamp(min=1e-8), torch.full_like(angle, 2.0))
    return q[:, 1:] * scale[:, None]


# Frozen expert.


class FrozenExpert(nn.Module):
    """Deterministic RL-Games continuous_a2c_logstd actor rebuilt from a checkpoint.

    Matches ``player.get_action(obs, is_deterministic=True)``: running-mean-std input
    normalization clipped to +-5, ELU MLP, linear mu head, then clamp to [-1, 1].
    """

    def __init__(self, checkpoint: Path, expected_sha256: str | None = None):
        super().__init__()
        checkpoint = Path(checkpoint).expanduser().resolve()
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = sha256(checkpoint)
        if expected_sha256 is not None and self.checkpoint_sha256 != expected_sha256:
            raise ValueError(f"Expert checkpoint hash {self.checkpoint_sha256} != expected {expected_sha256}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]

        self.register_buffer("obs_mean", state["running_mean_std.running_mean"].float().clone())
        self.register_buffer("obs_var", state["running_mean_std.running_var"].float().clone())
        layer_ids = sorted(
            {int(m.group(1)) for key in state if (m := re.fullmatch(r"a2c_network\.actor_mlp\.(\d+)\.weight", key))}
        )
        if not layer_ids:
            raise KeyError("No a2c_network.actor_mlp.*.weight entries in checkpoint")
        layers = []
        for layer_id in layer_ids:
            weight = state[f"a2c_network.actor_mlp.{layer_id}.weight"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(state[f"a2c_network.actor_mlp.{layer_id}.bias"])
            layers.append(linear)
        self.hidden = nn.ModuleList(layers)
        mu_weight = state["a2c_network.mu.weight"]
        self.mu = nn.Linear(mu_weight.shape[1], mu_weight.shape[0])
        self.mu.weight.data.copy_(mu_weight)
        self.mu.bias.data.copy_(state["a2c_network.mu.bias"])
        if self.obs_mean.shape != (OBS_DIM,) or self.mu.out_features != ACTION_DIM:
            raise ValueError(f"Unexpected expert shapes: obs {tuple(self.obs_mean.shape)}, action {self.mu.out_features}")
        self.layer_sizes = [layer.out_features for layer in layers]
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + EXPERT_OBS_EPSILON)
        x = torch.clamp(x, -EXPERT_OBS_NORM_CLIP, EXPERT_OBS_NORM_CLIP)
        for layer in self.hidden:
            x = nn.functional.elu(layer(x))
        return torch.clamp(self.mu(x), -1.0, 1.0)


# Intent specification and module.


@dataclass(frozen=True)
class IntentSpec:
    name: str
    kind: str  # blind | hand | joint | wrist | wrist_finger | obj | goal
    k: int = 0
    dim: int = 0

    @property
    def needs_expert(self) -> bool:
        return self.kind in ("hand", "joint", "wrist", "wrist_finger")

    @property
    def needs_lookahead(self) -> bool:
        return self.kind == "obj" and self.k in (3, 4)


def parse_intent(name: str) -> IntentSpec:
    name = name.strip().lower()
    if name == "blind":
        return IntentSpec(name, "blind", 0, 0)
    if name == "wrist-2":
        return IntentSpec(name, "wrist", 2, 2)
    if m := re.fullmatch(r"hand-(\d+)", name):
        k = int(m.group(1))
        if not 1 <= k <= ACTION_DIM:
            raise ValueError(f"hand-k needs 1 <= k <= {ACTION_DIM}: {name}")
        return IntentSpec(name, "hand", k, k)
    if m := re.fullmatch(r"joint-(\d+)", name):
        k = int(m.group(1))
        if not 1 <= k <= ACTION_DIM:
            raise ValueError(f"joint-k needs 1 <= k <= {ACTION_DIM}: {name}")
        return IntentSpec(name, "joint", k, k)
    if m := re.fullmatch(r"wrist2-finger-(\d+)", name):
        k = int(m.group(1))
        if not 1 <= k <= ACTION_DIM - 2:
            raise ValueError(f"wrist2-finger-k needs 1 <= k <= 18: {name}")
        return IntentSpec(name, "wrist_finger", k, 2 + k)
    if name == "goal-4":
        return IntentSpec(name, "goal", 4, 4)
    if name == "goal-yaw":
        return IntentSpec(name, "goal", 2, 2)
    if m := re.fullmatch(r"obj-([134])", name):
        k = int(m.group(1))
        return IntentSpec(name, "obj", k, k)
    raise ValueError(f"Unknown intent: {name}")


class IntentModule(nn.Module):
    def __init__(
        self,
        spec: IntentSpec,
        expert: FrozenExpert | None = None,
        pca_basis: Path | None = None,
        obj_lookahead: float | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.expert = expert
        uses_pca = spec.kind in ("hand", "joint", "wrist_finger")
        self.pca_basis = str(pca_basis) if uses_pca and pca_basis is not None else None
        self.pca_basis_sha256 = sha256(pca_basis) if uses_pca and pca_basis is not None else None
        self.obj_lookahead = obj_lookahead
        if spec.needs_expert and expert is None:
            raise ValueError(f"{spec.name} needs a frozen expert")
        if spec.needs_lookahead and (obj_lookahead is None or obj_lookahead <= 0):
            raise ValueError(f"{spec.name} needs a positive obj_lookahead (radians)")

        if uses_pca:
            if pca_basis is None:
                raise ValueError(f"{spec.name} needs a PCA basis")
            basis = np.load(pca_basis)
            if spec.kind == "hand":
                mu, w = basis["mu_robot"], basis["W_robot"][:, : spec.k]
            elif spec.kind == "joint":
                # Separate keys, so an action-PCA file cannot be passed to joint-K by mistake or vice versa.
                mu, w = basis["mu_joint"], basis["W_joint"][:, : spec.k]
            else:
                mu, w = basis["mu_finger"], basis["W_finger"][:, : spec.k]
            self.register_buffer("pca_mu", torch.as_tensor(mu, dtype=torch.float32))
            self.register_buffer("pca_w", torch.as_tensor(np.ascontiguousarray(w), dtype=torch.float32))

    @property
    def dim(self) -> int:
        return self.spec.dim

    def describe(self) -> dict:
        return {
            "intent": self.spec.name,
            "kind": self.spec.kind,
            "k": self.spec.k,
            "dim": self.spec.dim,
            "expert_checkpoint": self.expert.checkpoint if self.expert is not None else None,
            "expert_checkpoint_sha256": self.expert.checkpoint_sha256 if self.expert is not None else None,
            "pca_basis": self.pca_basis,
            "pca_basis_sha256": self.pca_basis_sha256,
            "obj_lookahead_rad": self.obj_lookahead if self.spec.needs_lookahead else None,
        }

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
            raise ValueError(f"Intent expects [N, {OBS_DIM}] observations, got {tuple(obs.shape)}")
        kind = self.spec.kind
        if kind == "blind":
            return obs.new_zeros((obs.shape[0], 0))
        if kind == "obj":
            return self._object_intent(obs)
        if kind == "goal":
            if self.spec.k == 2:
                return goal_yaw(obs[:, GOAL_ROT])
            return canonical_quat(obs[:, GOAL_ROT])

        action = self.expert(obs)
        if kind == "wrist":
            return action[:, WRIST]
        if kind in ("hand", "joint"):
            return (action - self.pca_mu) @ self.pca_w
        return torch.cat([action[:, WRIST], (action[:, FINGERS] - self.pca_mu) @ self.pca_w], dim=1)

    def _object_intent(self, obs: torch.Tensor) -> torch.Tensor:
        object_rot = obs[:, OBJECT_ROT]
        goal_rot = obs[:, GOAL_ROT]
        theta = rotation_distance(object_rot, goal_rot)
        if self.spec.k == 1:
            return theta[:, None]
        fraction = torch.clamp(self.obj_lookahead / theta.clamp(min=1e-6), max=1.0)
        subgoal = quat_slerp(object_rot, goal_rot, fraction)
        if self.spec.k == 4:
            return canonical_quat(subgoal)
        return quat_to_rotvec(quat_mul(subgoal, quat_conjugate(object_rot)))


def goal_yaw(q: torch.Tensor) -> torch.Tensor:
    """(sin psi, cos psi) of the rotation angle psi = 2 atan2(q_z, q_w) about world z; q and -q agree."""
    psi = 2.0 * torch.atan2(q[:, 3], q[:, 0])
    return torch.stack([torch.sin(psi), torch.cos(psi)], dim=1)


def quantise(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Clips to [-1, 1] and rounds each component to the nearest of 2**bits evenly spaced levels (endpoints included)."""
    levels = 2**bits - 1
    return torch.round((x.clamp(-1.0, 1.0) + 1.0) * (levels / 2.0)) * (2.0 / levels) - 1.0


class IntentChannel:
    """Per-environment channel between intent source and actor (rate sweep). Defaults pass the intent through.

    * ``hold`` = N: zero-order hold; the intent is recomputed when the episode step is a multiple of N
      (always at episode start) and held otherwise. A goal change inside an episode does not refresh it.
    * ``delay`` = d: the actor sees the intent computed d steps earlier in the same episode; until d steps
      have passed it sees the episode's first intent.
    * ``bits`` = b: each component is clipped to [-1, 1] and rounded to 2**b levels.

    At most one of them may be active. ``step`` is the env's episode step counter (0 on the first
    observation of an episode); call ``reset`` whenever all environments are reset outside of ``step``.
    """

    def __init__(self, hold: int = 1, delay: int = 0, bits: int = 0):
        if hold < 1 or delay < 0 or bits < 0:
            raise ValueError(f"Invalid intent channel: hold={hold} delay={delay} bits={bits}")
        if sum((hold > 1, delay > 0, bits > 0)) > 1:
            raise ValueError(f"Only one of hold, delay, bits may be active: hold={hold} delay={delay} bits={bits}")
        self.hold, self.delay, self.bits = int(hold), int(delay), int(bits)
        self.reset()

    @property
    def passthrough(self) -> bool:
        return self.hold == 1 and self.delay == 0 and self.bits == 0

    def reset(self) -> None:
        self._held: torch.Tensor | None = None
        self._buffer: torch.Tensor | None = None

    def __call__(self, intent: torch.Tensor, step: torch.Tensor | None) -> torch.Tensor:
        if self.passthrough or intent.shape[1] == 0:
            return intent
        if self.bits:
            return quantise(intent, self.bits)
        step = step.to(intent.device).long()
        if self.hold > 1:
            refresh = (step % self.hold) == 0
            if self._held is None or self._held.shape != intent.shape:
                refresh = torch.ones_like(refresh)
                self._held = intent.clone()
            self._held = torch.where(refresh.unsqueeze(1), intent, self._held)
            return self._held
        # Delay: ring buffer indexed by episode step; slot s % (d + 1) holds the intent of step s.
        size = self.delay + 1
        if self._buffer is None or self._buffer.shape[1:] != intent.shape:
            self._buffer = intent.unsqueeze(0).repeat(size, 1, 1)
        envs = torch.arange(intent.shape[0], device=intent.device)
        self._buffer[step % size, envs] = intent
        return self._buffer[(step - self.delay).clamp(min=0) % size, envs].clone()


def hide_goal(obs: torch.Tensor) -> torch.Tensor:
    """Removes the goal fields from [N, 157] observations, giving the [N, 149] goal-hidden actor input."""
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"hide_goal expects [N, {OBS_DIM}] observations, got {tuple(obs.shape)}")
    return torch.cat([obs[:, : GOAL_FIELDS.start], obs[:, GOAL_FIELDS.stop :]], dim=1)


def build_intent(
    name: str,
    expert_checkpoint: Path | None = None,
    expert_sha256: str | None = None,
    pca_basis: Path | None = None,
    obj_lookahead: float | None = None,
) -> IntentModule:
    spec = parse_intent(name)
    expert = FrozenExpert(expert_checkpoint, expert_sha256) if spec.needs_expert else None
    return IntentModule(spec, expert=expert, pca_basis=pca_basis, obj_lookahead=obj_lookahead)
