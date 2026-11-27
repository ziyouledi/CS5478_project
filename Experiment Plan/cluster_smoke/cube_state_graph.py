#!/usr/bin/env python3
"""Discrete 96-state cube geometry (24 cube orientations x 4 hinge detents) and BFS shortest-path planner.

Pure Python + NumPy (no Isaac Lab / torch dependency), so it can be unit-tested locally on CPU and imported by
both the A0/P2 Isaac Lab evaluation harness and the P2/P3 VLM text/vision RL trainers.

Conventions (matching z1_render_env.py and z1_build_mcq.py):
  - Cube base link frame axes: +x = red, +y = green, -x = yellow, -y = blue, +z = white (top layer), -z = magenta.
  - World/env frame: +z is UP (away from palm), +x/+y lie in the horizontal plane.
  - A discrete orientation o in {0..23} is uniquely identified by an integer 3x3 rotation matrix R in SO(3) with
    entries in {-1, 0, 1} (the columns of R are the cube's local x, y, z axes expressed in the env frame), or
    equivalently by the pair (up_face, front_face) where:
      * up_face    in {px, nx, py, ny, pz, nz} = cube face pointing along env +z (dot(R[:, axis], +z) == +1)
      * front_face in {px, nx, py, ny, pz, nz} = cube face pointing along env +x (dot(R[:, axis], +x) == +1)
  - A discrete detent d in {0, 1, 2, 3} corresponds to hinge angle d * (pi / 2) wrapped to (-pi, pi]:
      d=0 -> 0 deg, d=1 -> +90 deg, d=2 -> 180 deg, d=3 -> -90 deg.
  - Legal high-level actions from state (o, d):
      1. ("reorient", o_next) for any o_next in {0..23} (o_next != o): sets cube orientation to o_next, keeps d fixed.
      2. ("turn", +1) or ("turn", -1): ONLY legal when top layer points up (up_face(o) == "pz"); changes d to
         (d + delta) % 4, keeps o fixed.
    (Matches GOAL_LEVEL_RL_PLAN.md: Unified v4 was only trained to execute `turn` when top layer faces up.)
"""

from __future__ import annotations

import collections
import math
from typing import Iterable

import numpy as np

FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")
FACE_COLOUR = {"px": "red", "py": "green", "nx": "yellow", "ny": "blue", "pz": "white", "nz": "magenta"}
COLOUR_TO_FACE = {v: k for k, v in FACE_COLOUR.items()}
FACE_VECTORS = {
    "px": np.array([1, 0, 0], dtype=int),
    "nx": np.array([-1, 0, 0], dtype=int),
    "py": np.array([0, 1, 0], dtype=int),
    "ny": np.array([0, -1, 0], dtype=int),
    "pz": np.array([0, 0, 1], dtype=int),
    "nz": np.array([0, 0, -1], dtype=int),
}
SIDES = ("px", "py", "nx", "ny")  # counter-clockwise seen from +z


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 proper rotation matrix R to a canonical unit quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    # Canonical sign: first non-zero component positive
    for v in q:
        if abs(v) > 1e-9:
            if v < 0:
                q = -q
            break
    return q


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _build_orientations():
    """Enumerate all 24 axis-aligned cube orientations in deterministic order."""
    orientations = []
    # For each local face pointing +z (up) and each orthogonal local face pointing +x (front):
    # Note: R^T @ [0,0,1] = local vector pointing +z (row 2 of R).
    #       R^T @ [1,0,0] = local vector pointing +x (row 0 of R).
    #       R^T @ [0,1,0] = local vector pointing +y (row 1 of R) = cross(up_local, front_local).
    for up_name in ("pz", "nz", "px", "py", "nx", "ny"):
        row2 = FACE_VECTORS[up_name]
        for front_name in ("px", "py", "nx", "ny", "pz", "nz"):
            row0 = FACE_VECTORS[front_name]
            if int(np.dot(row0, row2)) != 0:
                continue
            row1 = np.cross(row2, row0)
            R = np.vstack([row0, row1, row2]).astype(int)
            assert int(round(np.linalg.det(R))) == 1
            orientations.append({
                "id": len(orientations),
                "up_face": up_name,
                "front_face": front_name,
                "up_colour": FACE_COLOUR[up_name],
                "front_colour": FACE_COLOUR[front_name],
                "R": R.astype(np.float64),
                "quat_wxyz": rotmat_to_quat_wxyz(R),
                "is_top_up": up_name == "pz",
            })
    assert len(orientations) == 24
    return orientations


ORIENTATIONS = _build_orientations()
ORIENT_QUATS = np.stack([o["quat_wxyz"] for o in ORIENTATIONS], axis=0)  # (24, 4)
ORIENT_MATS = np.stack([o["R"] for o in ORIENTATIONS], axis=0)          # (24, 3, 3)
PAIR_TO_ORIENT = {(o["up_face"], o["front_face"]): o["id"] for o in ORIENTATIONS}
COLOUR_PAIR_TO_ORIENT = {(o["up_colour"], o["front_colour"]): o["id"] for o in ORIENTATIONS}
TOP_UP_ORIENT_IDS = tuple(o["id"] for o in ORIENTATIONS if o["is_top_up"])


def detent_to_rad(detent: int) -> float:
    """Discrete detent in {0,1,2,3} -> wrapped angle in (-pi, pi]."""
    d = int(detent) % 4
    return (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi)[d]


def rad_to_detent(angle_rad: float) -> tuple[int, float]:
    """Continuous hinge angle (rad) -> (nearest detent in {0..3}, signed residual in deg)."""
    q = round(angle_rad / (0.5 * math.pi))
    res_rad = angle_rad - q * (0.5 * math.pi)
    res_rad = math.atan2(math.sin(res_rad), math.cos(res_rad))
    return int(q) % 4, math.degrees(res_rad)


def nearest_orientation(quat_wxyz: np.ndarray) -> tuple[int, float]:
    """Continuous unit quaternion (w, x, y, z) -> (nearest orientation id in {0..23}, angular error in deg)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    dots = np.abs(ORIENT_QUATS @ q).clip(0.0, 1.0)
    idx = int(np.argmax(dots))
    err_deg = math.degrees(2.0 * math.acos(float(dots[idx])))
    return idx, err_deg


def state_to_index(orient_id: int, detent: int) -> int:
    return int(orient_id) * 4 + (int(detent) % 4)


def index_to_state(state_idx: int) -> tuple[int, int]:
    return divmod(int(state_idx), 4)


def state_to_dict(orient_id: int, detent: int) -> dict:
    o = ORIENTATIONS[int(orient_id)]
    d = int(detent) % 4
    # For each base side colour, which top-layer strip colour sits on it at detent d?
    # SIDES = ("px", "py", "nx", "ny") = (red, green, yellow, blue) counter-clockwise seen from +z.
    # At detent d, base side SIDES[i] has top strip SIDES[(i - d) % 4].
    strips_by_base = {
        FACE_COLOUR[s]: FACE_COLOUR[SIDES[(i - d) % 4]]
        for i, s in enumerate(SIDES)
    }
    return {
        "orient_id": o["id"],
        "up_colour": o["up_colour"],
        "front_colour": o["front_colour"],
        "detent": d,
        "top_band_on_red": "none" if d == 0 else FACE_COLOUR[SIDES[(-d) % 4]],
        "strips_by_base": strips_by_base,
    }


# Elementary 90-degree world-axis rotations (Lever 2: SO(3) non-commutative step-by-step reorientation)
# Right-hand rule about world +x (roll), +y (pitch), +z (yaw):
ELEMENTARY_ROT_MATS = {
    "roll:+90":  np.array([[1, 0,  0], [0, 0, -1], [0, 1, 0]], dtype=int),
    "roll:-90":  np.array([[1, 0,  0], [0, 0,  1], [0, -1, 0]], dtype=int),
    "pitch:+90": np.array([[0, 0,  1], [0, 1,  0], [-1, 0, 0]], dtype=int),
    "pitch:-90": np.array([[0, 0, -1], [0, 1,  0], [1, 0, 0]], dtype=int),
    "yaw:+90":   np.array([[0, -1, 0], [1, 0,  0], [0, 0, 1]], dtype=int),
    "yaw:-90":   np.array([[0,  1, 0], [-1, 0, 0], [0, 0, 1]], dtype=int),
}


def _build_elementary_transitions() -> dict[tuple[int, str], int]:
    """Map (orient_id, rot_cmd) -> next_orient_id after applying world-frame 90-deg rotation R_world @ R_cube."""
    trans = {}
    for o in ORIENTATIONS:
        R = o["R"].astype(int)
        for cmd, W in ELEMENTARY_ROT_MATS.items():
            R_next = W @ R
            # row 2 of R_next is local face pointing +z; row 0 is local face pointing +x
            up_next = next(k for k, v in FACE_VECTORS.items() if np.array_equal(v, R_next[2]))
            front_next = next(k for k, v in FACE_VECTORS.items() if np.array_equal(v, R_next[0]))
            trans[(o["id"], cmd)] = PAIR_TO_ORIENT[(up_next, front_next)]
    return trans


ELEMENTARY_TRANS = _build_elementary_transitions()


def _precompute_local90_bfs():
    """Precompute exact shortest-path distance table (96, 96) when reorient is restricted to +-90 deg world axes."""
    dist = np.full((96, 96), -1, dtype=int)
    for target in range(96):
        dist[target, target] = 0
        q = collections.deque([target])
        while q:
            cur = q.popleft()
            co, cd = index_to_state(cur)
            # Predecessors under elementary rotations (inverse of +90 is -90)
            preds = [state_to_index(ELEMENTARY_TRANS[(co, cmd)], cd) for cmd in ELEMENTARY_ROT_MATS]
            if ORIENTATIONS[co]["is_top_up"]:
                preds.append(state_to_index(co, (cd - 1) % 4))
                preds.append(state_to_index(co, (cd + 1) % 4))
            for p in preds:
                if dist[p, target] == -1:
                    dist[p, target] = dist[cur, target] + 1
                    q.append(p)
    assert (dist >= 0).all(), "Local-90 96-state graph must be strongly connected"
    return dist


DIST_TABLE_LOCAL90 = _precompute_local90_bfs()


def legal_actions_local90(orient_id: int, detent: int) -> list[tuple[str, str]]:
    """All legal actions under Lever 2 (elementary 90-deg rotations + top-up turns)."""
    actions = [("rotate", cmd) for cmd in ELEMENTARY_ROT_MATS]
    if ORIENTATIONS[orient_id]["is_top_up"]:
        actions.append(("turn", "+90"))
        actions.append(("turn", "-90"))
    return actions


def step_symbolic_local90(orient_id: int, detent: int, action: tuple[str, str]) -> tuple[int, int, bool]:
    """Apply an elementary 90-deg rotation or turn action symbolically."""
    kind, cmd = action
    if kind == "rotate" and cmd in ELEMENTARY_ROT_MATS:
        return ELEMENTARY_TRANS[(int(orient_id), cmd)], int(detent) % 4, True
    if kind == "turn" and cmd in ("+90", "-90"):
        if ORIENTATIONS[int(orient_id)]["is_top_up"]:
            delta = +1 if cmd == "+90" else -1
            return int(orient_id), (int(detent) + delta) % 4, True
        return int(orient_id), int(detent) % 4, False
    return int(orient_id), int(detent) % 4, False


def optimal_actions_local90(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[str, str]]:
    if src == dst:
        return []
    d0 = int(DIST_TABLE_LOCAL90[state_to_index(*src), state_to_index(*dst)])
    dst_idx = state_to_index(*dst)
    best = []
    for act in legal_actions_local90(*src):
        no, nd, ok = step_symbolic_local90(*src, act)
        if ok and DIST_TABLE_LOCAL90[state_to_index(no, nd), dst_idx] == d0 - 1:
            best.append(act)
    return best


def sample_stratified_tasks_local90(
    n_per_level: int = 50, levels: Iterable[int] = (1, 2, 3, 4), seed: int = 20261120
) -> list[dict]:
    """Sample balanced (src, dst) tasks stratified by optimal BFS length L in DIST_TABLE_LOCAL90."""
    import random

    rng = random.Random(seed)
    tasks = []
    for L in levels:
        pairs = [(index_to_state(s), index_to_state(t))
                 for s in range(96) for t in range(96) if DIST_TABLE_LOCAL90[s, t] == L]
        rng.shuffle(pairs)
        for idx, (src, dst) in enumerate(pairs[:n_per_level]):
            tasks.append({
                "task_id": f"L{L}_{idx:03d}",
                "optimal_steps": L,
                "src": {"orient_id": src[0], "detent": src[1]},
                "dst": {"orient_id": dst[0], "detent": dst[1]},
            })
    return tasks


def legal_actions(orient_id: int, detent: int) -> list[tuple[str, int]]:
    """All legal high-level actions from (orient_id, detent)."""
    actions = [("reorient", o) for o in range(24) if o != orient_id]
    if ORIENTATIONS[orient_id]["is_top_up"]:
        actions.append(("turn", +1))
        actions.append(("turn", -1))
    return actions


def step_symbolic(orient_id: int, detent: int, action: tuple[str, int]) -> tuple[int, int, bool]:
    """Apply a high-level action symbolically. Returns (next_o, next_d, is_legal)."""
    kind, val = action
    if kind == "reorient":
        if 0 <= int(val) < 24:
            return int(val), int(detent) % 4, True
        return orient_id, int(detent) % 4, False
    if kind == "turn":
        if ORIENTATIONS[orient_id]["is_top_up"] and int(val) in (-1, +1):
            return orient_id, (int(detent) + int(val)) % 4, True
        return orient_id, int(detent) % 4, False
    return orient_id, int(detent) % 4, False


def _precompute_all_pairs_bfs():
    """Precompute exact shortest-path distance table (96, 96) and optimal first actions."""
    dist = np.full((96, 96), -1, dtype=int)
    for target in range(96):
        dist[target, target] = 0
        q = collections.deque([target])
        while q:
            cur = q.popleft()
            co, cd = index_to_state(cur)
            # Predecessors of (co, cd):
            preds = [state_to_index(po, cd) for po in range(24) if po != co]
            if ORIENTATIONS[co]["is_top_up"]:
                preds.append(state_to_index(co, (cd - 1) % 4))
                preds.append(state_to_index(co, (cd + 1) % 4))
            for p in preds:
                if dist[p, target] == -1:
                    dist[p, target] = dist[cur, target] + 1
                    q.append(p)
    assert (dist >= 0).all(), "96-state graph must be strongly connected"
    return dist


DIST_TABLE = _precompute_all_pairs_bfs()


def shortest_distance(src: tuple[int, int], dst: tuple[int, int]) -> int:
    return int(DIST_TABLE[state_to_index(*src), state_to_index(*dst)])


def optimal_actions(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[str, int]]:
    """Return all legal actions from src that strictly decrease the BFS distance to dst."""
    if src == dst:
        return []
    d0 = shortest_distance(src, dst)
    dst_idx = state_to_index(*dst)
    best = []
    for act in legal_actions(*src):
        no, nd, ok = step_symbolic(*src, act)
        if ok and DIST_TABLE[state_to_index(no, nd), dst_idx] == d0 - 1:
            best.append(act)
    return best


def bfs_plan(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[str, int]]:
    """Return one deterministic shortest legal action sequence from src to dst.

    Tie-breaking preference:
      1. If a `turn` is optimal, prefer `turn` (fewer reorientations).
      2. If reorienting to a top-up orientation before turning, prefer orient_id=0 (white up, red front) or
         the smallest orient_id so behaviour is deterministic.
    """
    cur = src
    plan = []
    while cur != dst:
        opts = optimal_actions(cur, dst)
        assert opts, f"No optimal action from {cur} to {dst}"
        opts.sort(key=lambda a: (0 if a[0] == "turn" else 1, a[1]))
        act = opts[0]
        plan.append(act)
        no, nd, _ = step_symbolic(*cur, act)
        cur = (no, nd)
    return plan


def sample_stratified_tasks(n_per_level: int = 50, levels: Iterable[int] = (1, 2, 3, 4), seed: int = 20261120) -> list[dict]:
    """Sample balanced (src, dst) tasks stratified by optimal BFS length L in `levels`."""
    import random

    rng = random.Random(seed)
    tasks = []
    for L in levels:
        pairs = [(index_to_state(s), index_to_state(t))
                 for s in range(96) for t in range(96) if DIST_TABLE[s, t] == L]
        rng.shuffle(pairs)
        picked = pairs[:n_per_level]
        for idx, (src, dst) in enumerate(picked):
            tasks.append({
                "task_id": f"L{L}_{idx:03d}",
                "optimal_steps": L,
                "src": {"orient_id": src[0], "detent": src[1]},
                "dst": {"orient_id": dst[0], "detent": dst[1]},
                "reference_plan": [{"kind": k, "value": v} for k, v in bfs_plan(src, dst)],
            })
    return tasks


def _self_test() -> None:
    # 1. All 24 orientations are distinct proper rotations and round-trip through quaternions
    for o in ORIENTATIONS:
        R_back = quat_wxyz_to_rotmat(o["quat_wxyz"])
        assert np.allclose(o["R"], R_back, atol=1e-6)
        oid, err = nearest_orientation(o["quat_wxyz"])
        assert oid == o["id"] and err < 1e-5
    assert len(TOP_UP_ORIENT_IDS) == 4
    # 2. Check graph diameter and distance distribution
    counts = collections.Counter(int(x) for x in DIST_TABLE.flatten())
    assert max(counts) == 4, f"Expected diameter 4, got {max(counts)}"
    # 3. Verify every sampled plan reaches dst legally in exactly L steps
    tasks = sample_stratified_tasks(n_per_level=20, levels=(1, 2, 3, 4))
    for t in tasks:
        cur = (t["src"]["orient_id"], t["src"]["detent"])
        dst = (t["dst"]["orient_id"], t["dst"]["detent"])
        plan = [(a["kind"], a["value"]) for a in t["reference_plan"]]
        assert len(plan) == t["optimal_steps"]
        for act in plan:
            no, nd, ok = step_symbolic(*cur, act)
            assert ok, f"Illegal action {act} from {cur}"
            cur = (no, nd)
        assert cur == dst
    print(f"cube_state_graph_test=PASS diameter={max(counts)} dist_counts={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    _self_test()
