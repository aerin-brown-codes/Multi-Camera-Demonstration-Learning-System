from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

PARALLEL_DIRECTIONS_BELOW = 1e-6
JOINT_RANGE_ROUNDING_RAD = 1e-6
ONLY_REPORT_PENETRATION = 0.0


@dataclass(frozen=True)
class ObjectPose:
    xy: np.ndarray
    z_or_none_for_resting: float | None
    yaw_rad: float
    roll_pitch_rad: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class HandPose:
    T_world_palm: np.ndarray
    joint_angles_rad: dict[str, float]


@dataclass(frozen=True)
class Pose:
    name: str
    objects: dict[str, ObjectPose]
    hands: dict[str, HandPose]
    together: tuple[str, ...] = ()


@dataclass(frozen=True)
class HandBodies:
    mocap_body: str
    palm_body: str
    joint_prefix: str


@dataclass(frozen=True)
class ScenePoses:
    object_bodies: dict[str, str]
    hand_bodies: dict[str, HandBodies]
    poses: list[Pose]
    wrist_joints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Overlap:
    owners: tuple[str, str]
    parts: tuple[str, str]
    depth_m: float

    def __str__(self) -> str:
        return (f"{self.owners[0]} ({self.parts[0]}) is {self.depth_m * 1000:.1f} mm "
                f"inside {self.owners[1]} ({self.parts[1]})")


class PoseOverlap(ValueError):
    pass


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def R_world_palm_with_fingers_along_z_and_palm_facing_minus_y(fingers_point, palm_faces) -> np.ndarray:
    fingers = unit(fingers_point)
    facing = np.asarray(palm_faces, float)
    facing = facing - facing.dot(fingers) * fingers
    if np.linalg.norm(facing) < PARALLEL_DIRECTIONS_BELOW:
        raise ValueError(f"palm_faces {list(palm_faces)} is parallel to fingers_point {list(fingers_point)}")
    back_of_hand = -unit(facing)
    return np.column_stack([np.cross(back_of_hand, fingers), back_of_hand, fingers])


def T_world_palm(palm_pos, fingers_point, palm_faces) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_world_palm_with_fingers_along_z_and_palm_facing_minus_y(fingers_point, palm_faces)
    T[:3, 3] = palm_pos
    return T


def load_scene_poses(path) -> ScenePoses:
    cfg = yaml.safe_load(Path(path).read_text())
    shapes = cfg.get("finger_shapes", {})
    objects, hands = cfg.get("objects", {}), cfg.get("hands", {})
    wrist_joints = cfg.get("wrist_joints", {})

    def hand_pose(pose_name, h):
        if h["fingers"] not in shapes:
            raise ValueError(f"pose '{pose_name}': no finger shape named '{h['fingers']}'")
        angles = dict(shapes[h["fingers"]])
        for setting, degrees in h.get("wrist", {}).items():
            if setting not in wrist_joints:
                raise ValueError(f"pose '{pose_name}': wrist has no setting '{setting}', only {sorted(wrist_joints)}")
            if wrist_joints[setting] in angles:
                raise ValueError(f"pose '{pose_name}': finger shape '{h['fingers']}' already sets the wrist joint "
                                 f"{wrist_joints[setting]}")
            angles[wrist_joints[setting]] = float(np.deg2rad(degrees))
        return HandPose(T_world_palm(h["palm_pos"], h["fingers_point"], h["palm_faces"]), angles)

    def object_pose(o):
        pos = o["pos"]
        roll_deg, pitch_deg = o.get("tilt_deg", (0.0, 0.0))
        return ObjectPose(np.array(pos[:2], float), float(pos[2]) if len(pos) == 3 else None,
                          np.deg2rad(o.get("yaw_deg", 0.0)), (np.deg2rad(roll_deg), np.deg2rad(pitch_deg)))

    def moved_together(p):
        unknown = set(p.get("together", [])) - objects.keys() - hands.keys()
        if unknown:
            raise ValueError(f"pose '{p['name']}': together names {sorted(unknown)}, which the scene does not have")
        return tuple(p.get("together", []))

    poses = [
        Pose(
            name=p["name"],
            objects={name: object_pose(p[name]) for name in objects if name in p},
            hands={name: hand_pose(p["name"], p[name]) for name in hands if name in p},
            together=moved_together(p),
        )
        for p in cfg.get("poses", [])
    ]
    return ScenePoses(
        object_bodies=dict(objects),
        hand_bodies={name: HandBodies(**h) for name, h in hands.items()},
        poses=poses,
        wrist_joints=dict(wrist_joints),
    )


def held_pose_index(frame: int, fps: float, hold_s: float, pose_count: int) -> int:
    frames_per_pose = max(1, round(hold_s * fps))
    return (frame // frames_per_pose) % pose_count


def T_of_body(data: mujoco.MjData, body_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def quat_wxyz(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def R_about_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_world_object_turned_about_z_after_pitch_after_roll(o: ObjectPose) -> np.ndarray:
    (cr, sr), (cp, sp) = [(np.cos(a), np.sin(a)) for a in o.roll_pitch_rad]
    R_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return R_about_z(o.yaw_rad) @ R_pitch @ R_roll


class Poser:
    def __init__(self, model: mujoco.MjModel, scene: ScenePoses):
        self.model = model
        self._object_qpos = {}
        for name, body in scene.object_bodies.items():
            joint = model.body(body).jntadr[0]
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"object '{name}': body '{body}' needs a free joint to be placed")
            self._object_qpos[name] = int(model.jnt_qposadr[joint])
        self._hand_mocap = {name: int(model.body(h.mocap_body).mocapid[0]) for name, h in scene.hand_bodies.items()}
        self._hand_palm = {name: model.body(h.palm_body).id for name, h in scene.hand_bodies.items()}
        self._hand_joints = {
            name: {
                model.joint(j).name.removeprefix(h.joint_prefix): j
                for j in range(model.njnt) if model.joint(j).name.startswith(h.joint_prefix)
            }
            for name, h in scene.hand_bodies.items()
        }
        self._geom_owner, self._geom_part = self._name_geoms()
        self._pairs_mujoco_never_reports = self._pairs_of_geoms_that_cannot_move()

    def resting_z(self, object_name: str) -> float:
        return float(self.model.qpos0[self._object_qpos[object_name] + 2])

    def joint_ranges_rad(self, hand: str) -> dict[str, tuple[float, float]]:
        return {name: (float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1]))
                for name, j in self._hand_joints[hand].items()}

    def place(self, data: mujoco.MjData, pose: Pose) -> None:
        if pose.objects.keys() != self._object_qpos.keys() or pose.hands.keys() != self._hand_mocap.keys():
            raise ValueError(f"pose '{pose.name}' must place exactly "
                             f"{sorted(self._object_qpos) + sorted(self._hand_mocap)}, "
                             f"got {sorted(pose.objects) + sorted(pose.hands)}")
        for name, o in pose.objects.items():
            adr = self._object_qpos[name]
            z = self.resting_z(name) if o.z_or_none_for_resting is None else o.z_or_none_for_resting
            data.qpos[adr:adr + 3] = [*o.xy, z]
            data.qpos[adr + 3:adr + 7] = quat_wxyz(R_world_object_turned_about_z_after_pitch_after_roll(o))
        for name, h in pose.hands.items():
            self._write_joint_angles(data, pose.name, name, h.joint_angles_rad)
            data.mocap_pos[self._hand_mocap[name]] = 0
            data.mocap_quat[self._hand_mocap[name]] = [1, 0, 0, 0]
        mujoco.mj_kinematics(self.model, data)
        for name, h in pose.hands.items():
            T_mocap_palm = T_of_body(data, self._hand_palm[name])
            T_world_mocap = h.T_world_palm @ np.linalg.inv(T_mocap_palm)
            data.mocap_pos[self._hand_mocap[name]] = T_world_mocap[:3, 3]
            data.mocap_quat[self._hand_mocap[name]] = quat_wxyz(T_world_mocap[:3, :3])
        mujoco.mj_forward(self.model, data)

    def overlaps(self, data: mujoco.MjData, deeper_than_m: float) -> list[Overlap]:
        deepest = {}
        for g1, g2, dist in self._close_geoms_of_different_owners(data):
            owners = self._geom_owner[g1], self._geom_owner[g2]
            pair, depth_m = frozenset(owners), -dist
            if depth_m > deeper_than_m and (pair not in deepest or depth_m > deepest[pair].depth_m):
                deepest[pair] = Overlap(owners, (self._geom_part[g1], self._geom_part[g2]), depth_m)
        return sorted(deepest.values(), key=lambda o: -o.depth_m)

    def overlaps_of_a_thing_with_itself(self, data: mujoco.MjData, deeper_than_m: float) -> list[Overlap]:
        deepest = {}
        for c in data.contact[:data.ncon]:
            owner, depth_m = self._geom_owner[c.geom1], -c.dist
            if owner != self._geom_owner[c.geom2] or depth_m <= deeper_than_m:
                continue
            if owner not in deepest or depth_m > deepest[owner].depth_m:
                deepest[owner] = Overlap((owner, owner), (self._geom_part[c.geom1], self._geom_part[c.geom2]), depth_m)
        return sorted(deepest.values(), key=lambda o: -o.depth_m)

    def require_no_overlaps(self, data: mujoco.MjData, poses: list[Pose], allowed_overlap_m: float) -> None:
        for pose in poses:
            self.place(data, pose)
            overlaps = self.overlaps(data, allowed_overlap_m)
            if overlaps:
                raise PoseOverlap(f"pose '{pose.name}': " + "; ".join(str(o) for o in overlaps)
                                  + f" (allowed {allowed_overlap_m * 1000:g} mm)")

    def _write_joint_angles(self, data, pose_name, hand, angles_rad) -> None:
        joints = self._hand_joints[hand]
        unknown = angles_rad.keys() - joints.keys()
        if unknown:
            raise ValueError(f"pose '{pose_name}': hand '{hand}' has no joints {sorted(unknown)}")
        for joint_name, j in joints.items():
            angle = angles_rad.get(joint_name, 0.0)
            low, high = self.model.jnt_range[j]
            if self.model.jnt_limited[j] and not low - JOINT_RANGE_ROUNDING_RAD <= angle <= high + JOINT_RANGE_ROUNDING_RAD:
                raise ValueError(f"pose '{pose_name}': hand '{hand}' joint {joint_name} = {angle} rad "
                                 f"is outside its range [{low:.4f}, {high:.4f}]")
            data.qpos[self.model.jnt_qposadr[j]] = angle

    def _name_geoms(self) -> tuple[list[str], list[str]]:
        m = self.model
        owners, parts = [], []
        for g in range(m.ngeom):
            body = m.geom_bodyid[g]
            root = m.body_rootid[body]
            geom_name = m.geom(g).name
            owners.append(m.body(root).name if root else geom_name or "world")
            parts.append(geom_name or m.body(body).name)
        return owners, parts

    def _pairs_of_geoms_that_cannot_move(self) -> list[tuple[int, int]]:
        m = self.model
        cannot_move = m.body_dofnum[m.body_weldid[m.geom_bodyid]] == 0
        fixed = [g for g in np.flatnonzero(cannot_move) if m.geom_contype[g] or m.geom_conaffinity[g]]
        return [
            (int(a), int(b)) for i, a in enumerate(fixed) for b in fixed[i + 1:]
            if self._geom_owner[a] != self._geom_owner[b]
            and (m.geom_bodyid[a] or m.geom_bodyid[b])
            and (m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a])
        ]

    def _close_geoms_of_different_owners(self, data):
        for c in data.contact[:data.ncon]:
            if self._geom_owner[c.geom1] != self._geom_owner[c.geom2]:
                yield c.geom1, c.geom2, c.dist
        for a, b in self._pairs_mujoco_never_reports:
            yield a, b, mujoco.mj_geomDistance(self.model, data, a, b, ONLY_REPORT_PENETRATION, None)
