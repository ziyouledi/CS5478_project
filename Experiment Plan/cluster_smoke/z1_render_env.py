"""Unified hinged-cube task with Z1 rendering: coloured face plates and fixed per-env cameras.

Physics, reward, observation and goals are those of ShadowHandUnifiedEnv (v4 runs unchanged on it); only the
visual scene differs. GOAL_LEVEL_RL_PLAN.md "Z1 的渲染设定".

Cube visuals (visual-only prims under the two links, no collision / mass, as the video boxes):
  dark core boxes for the bottom part and the top layer, and on every outer face a plate covering
  PLATE_FRACTION of the face, so the split between the layers shows as a dark seam.
  Side colours are the same on the bottom part and the top layer, so at detent 0 every side face is one
  colour (like a solved Rubik's cube) and a quarter turn of the top layer shows as a colour offset.
  The top face is white with a black stripe from its centre towards +x of the top layer.
Each plate carries a semantic class (FACE_LABELS) for per-face visibility; the hand is labelled "hand".

Cameras (TiledCamera, one tile per env): Z1_CAMERAS = "name:azimuth_deg:elevation_deg:distance_m,..." about
Z1_TARGET ("x y z" in the env frame, default the stock in-hand cube position 0 -0.39 0.6). Azimuth is measured
in the env x-y plane from +x. The poses are set by pose_z1_cameras() after construction (fixed in the env frame).
Z1_IMAGE (pixels, square, default 512) and Z1_FOCAL (mm, horizontal aperture 20.955 mm, default 24).
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.assets import Articulation

from hinged_cube_env import ShadowHandUnifiedEnv


PLATE_FRACTION = 0.88
PLATE_THICKNESS = 0.001
HORIZONTAL_APERTURE_MM = 20.955
# Side colours by outward face direction of the link frame (same for both parts); bottom and top faces.
SIDE_COLOURS = {
    "px": (0.85, 0.08, 0.08),  # red
    "py": (0.10, 0.65, 0.20),  # green
    "nx": (0.95, 0.80, 0.05),  # yellow
    "ny": (0.10, 0.30, 0.90),  # blue
}
BOTTOM_COLOUR = (0.75, 0.15, 0.75)  # magenta, bottom part -z face
TOP_COLOUR = (0.95, 0.95, 0.95)  # white, top layer +z face
STRIPE_COLOUR = (0.02, 0.02, 0.02)
CORE_COLOUR = (0.08, 0.08, 0.08)
# Semantic classes; index 0 is reserved for "everything else" (ground, unlabelled) in the saved label maps.
FACE_LABELS = [
    "base_px", "base_py", "base_nx", "base_ny", "base_nz",
    "top_px", "top_py", "top_nx", "top_ny", "top_pz",
    "stripe", "core_base", "core_top", "hand",
]
CUBE_LABELS = [name for name in FACE_LABELS if name != "hand"]


def parse_cameras(spec: str) -> list[tuple[str, float, float, float]]:
    cameras = []
    for item in spec.split(","):
        name, az, el, dist = item.strip().split(":")
        cameras.append((name, float(az), float(el), float(dist)))
    if len({name for name, *_ in cameras}) != len(cameras):
        raise ValueError(f"Duplicate camera names in Z1_CAMERAS={spec}")
    return cameras


DEFAULT_CAMERAS = "c1:-50:38:0.33,c3:-50:85:0.33"


class ShadowHandUnifiedZ1Env(ShadowHandUnifiedEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.z1_cameras = parse_cameras(os.environ.get("Z1_CAMERAS", DEFAULT_CAMERAS))
        self.z1_target = tuple(float(v) for v in os.environ.get("Z1_TARGET", "0 -0.39 0.6").split())
        self.z1_image = int(os.environ.get("Z1_IMAGE", "512"))
        self.z1_focal = float(os.environ.get("Z1_FOCAL", "24"))
        cfg.robot_cfg.spawn.semantic_tags = [("class", "hand")]
        super().__init__(cfg, render_mode, **kwargs)

    def _face_plates(self) -> list[tuple[str, str, tuple, tuple, tuple]]:
        """(link, label, size, translation in the link frame, colour) for every visual prim of the cube."""
        sx, sy, sz = self.hinge_geometry["size"]
        f, t = PLATE_FRACTION, PLATE_THICKNESS
        base_h, top_h = sz * 2.0 / 3.0, sz / 3.0
        # Bottom part: link frame = cube centre, part spans z in [-sz/2, sz/6]. Top layer: link frame on the
        # split plane, part spans z in [0, sz/3].
        parts = (("base", base_h, -sz / 6.0), ("top", top_h, top_h / 2.0))
        plates = []
        for link, height, zc in parts:
            plates.append((link, f"core_{link}", (sx, sy, height), (0.0, 0.0, zc), CORE_COLOUR))
            plates.append((link, f"{link}_px", (t, f * sy, f * height), (sx / 2 + t / 2, 0.0, zc), SIDE_COLOURS["px"]))
            plates.append((link, f"{link}_nx", (t, f * sy, f * height), (-sx / 2 - t / 2, 0.0, zc), SIDE_COLOURS["nx"]))
            plates.append((link, f"{link}_py", (f * sx, t, f * height), (0.0, sy / 2 + t / 2, zc), SIDE_COLOURS["py"]))
            plates.append((link, f"{link}_ny", (f * sx, t, f * height), (0.0, -sy / 2 - t / 2, zc), SIDE_COLOURS["ny"]))
        plates.append(("base", "base_nz", (f * sx, f * sy, t), (0.0, 0.0, -sz / 2 - t / 2), BOTTOM_COLOUR))
        plates.append(("top", "top_pz", (f * sx, f * sy, t), (0.0, 0.0, top_h + t / 2), TOP_COLOUR))
        stripe_len = 0.4 * sx
        plates.append(
            ("top", "stripe", (stripe_len, sy / 8.0, t), (stripe_len / 2.0, 0.0, top_h + 1.5 * t), STRIPE_COLOUR)
        )
        return plates

    def _setup_scene(self):
        # As ShadowHandHingedCubeEnv._setup_scene, with face plates instead of the video boxes, plus cameras.
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        for index, (link, label, size, translation, colour) in enumerate(self._face_plates()):
            box_cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour, roughness=0.6),
                semantic_tags=[("class", label)],
            )
            box_cfg.func(f"/World/envs/env_0/object/{link}/z1_{index}_{label}", box_cfg, translation=translation)
        self.z1_sensors: dict[str, TiledCamera] = {}
        for name, *_ in self.z1_cameras:
            cam_cfg = TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/Z1Cam_{name}",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                data_types=["rgb", "semantic_segmentation"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self.z1_focal,
                    focus_distance=400.0,
                    horizontal_aperture=HORIZONTAL_APERTURE_MM,
                    clipping_range=(0.02, 50.0),
                ),
                width=self.z1_image,
                height=self.z1_image,
                colorize_semantic_segmentation=False,
                semantic_filter="class:*",
            )
            self.z1_sensors[name] = TiledCamera(cam_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["object"] = self.object
        for name, sensor in self.z1_sensors.items():
            self.scene.sensors[f"z1_{name}"] = sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def z1_eye(self, azimuth_deg: float, elevation_deg: float, distance: float) -> tuple[float, float, float]:
        az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
        tx, ty, tz = self.z1_target
        return (
            tx + distance * math.cos(el) * math.cos(az),
            ty + distance * math.cos(el) * math.sin(az),
            tz + distance * math.sin(el),
        )

    def pose_z1_cameras(self) -> dict[str, dict]:
        """Put every camera at its eye in each env, looking at Z1_TARGET; returns the poses in the env frame."""
        origins = self.scene.env_origins
        target = torch.tensor(self.z1_target, device=self.device).expand(self.num_envs, 3)
        poses = {}
        for name, az, el, dist in self.z1_cameras:
            eye = self.z1_eye(az, el, dist)
            eyes = torch.tensor(eye, device=self.device).expand(self.num_envs, 3)
            self.z1_sensors[name].set_world_poses_from_view(eyes + origins, target + origins)
            poses[name] = {"azimuth_deg": az, "elevation_deg": el, "distance_m": dist, "eye": list(eye)}
        return poses

    def z1_projection(self, name: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(eye in the env frame, rows = right / down / forward unit vectors, focal length in pixels) of a camera."""
        _, az, el, dist = next(c for c in self.z1_cameras if c[0] == name)
        eye = torch.tensor(self.z1_eye(az, el, dist), device=self.device)
        forward = torch.tensor(self.z1_target, device=self.device) - eye
        forward = forward / forward.norm()
        right = torch.linalg.cross(forward, torch.tensor((0.0, 0.0, 1.0), device=self.device))
        right = right / right.norm()
        down = torch.linalg.cross(forward, right)
        fx = self.z1_focal / HORIZONTAL_APERTURE_MM * self.z1_image
        return eye, torch.stack((right, down, forward)), fx
