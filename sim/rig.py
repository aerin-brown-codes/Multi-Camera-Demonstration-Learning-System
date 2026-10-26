from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

WORLD_UP = np.array([0.0, 0.0, 1.0])
IMAGE_UP_WHEN_LOOKING_STRAIGHT_UP_OR_DOWN = np.array([0.0, 1.0, 0.0])
OPENCV_TO_MUJOCO_CAMERA_AXES = np.array([1, -1, -1])


@dataclass(frozen=True)
class Profile:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_min_m: float
    depth_max_m: float
    depth_scale: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass(frozen=True)
class Camera:
    name: str
    profile: Profile
    position: np.ndarray
    R_world_cam: np.ndarray

    @property
    def T_world_cam(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R_world_cam
        T[:3, 3] = self.position
        return T


@dataclass(frozen=True)
class Rig:
    fps: float
    cameras: list[Camera]


def look_at(position, target) -> np.ndarray:
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, WORLD_UP)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, IMAGE_UP_WHEN_LOOKING_STRAIGHT_UP_OR_DOWN)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def load_rig(path) -> Rig:
    cfg = yaml.safe_load(Path(path).read_text())
    profiles = {name: Profile(**p) for name, p in cfg["profiles"].items()}
    cameras = [
        Camera(
            name=c["name"],
            profile=profiles[c["profile"]],
            position=np.array(c["position"], float),
            R_world_cam=look_at(c["position"], c["look_at"]),
        )
        for c in cfg["cameras"]
    ]
    return Rig(fps=float(cfg["fps"]), cameras=cameras)


def mujoco_quat(R_world_cam) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (np.asarray(R_world_cam) * OPENCV_TO_MUJOCO_CAMERA_AXES).flatten())
    return q


def opencv_to_mujoco_principal_pixel(p: Profile) -> list[float]:
    return [p.width / 2 - 0.5 - p.cx, p.height / 2 - 0.5 - p.cy]


def add_cameras(spec: mujoco.MjSpec, rig: Rig) -> None:
    for cam in rig.cameras:
        p = cam.profile
        c = spec.worldbody.add_camera(name=cam.name, pos=cam.position, quat=mujoco_quat(cam.R_world_cam))
        c.resolution = [p.width, p.height]
        c.sensor_size = [p.width, p.height]
        c.focal_pixel = [p.fx, p.fy]
        c.principal_pixel = opencv_to_mujoco_principal_pixel(p)
    g = spec.visual.global_
    g.offwidth = max([g.offwidth] + [c.profile.width for c in rig.cameras])
    g.offheight = max([g.offheight] + [c.profile.height for c in rig.cameras])
