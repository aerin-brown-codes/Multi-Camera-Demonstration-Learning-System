from dataclasses import dataclass
from functools import partial

import mujoco
import numpy as np

from sim.draws import chosen_by_share
from sim.poses import HandPose, ObjectPose, Pose, Poser, R_about_z, ScenePoses, T_world_palm
from sim.world import TableTop

MIRROR_LEFT_TO_RIGHT = np.diag([-1.0, 1.0, 1.0, 1.0])
SPACE_BETWEEN_THINGS_OUT_OF_VIEW_M = 1.0


@dataclass(frozen=True)
class VariedPose:
    pose: Pose
    kind: str
    made_from: str
    mirrored: bool


def mirrored_left_to_right(pose: Pose) -> Pose:
    hand_names = list(pose.hands)
    other_hand = dict(zip(hand_names, reversed(hand_names)))
    return Pose(
        pose.name,
        {name: ObjectPose(o.xy * [-1.0, 1.0], o.z_or_none_for_resting, -o.yaw_rad,
                          (o.roll_pitch_rad[0], -o.roll_pitch_rad[1]))
         for name, o in pose.objects.items()},
        {other_hand[name]: HandPose(MIRROR_LEFT_TO_RIGHT @ h.T_world_palm @ MIRROR_LEFT_TO_RIGHT, h.joint_angles_rad)
         for name, h in pose.hands.items()},
        tuple(other_hand.get(name, name) for name in pose.together),
    )


def T_turning_about_a_vertical_line_then_shifting(turn_rad: float, line_xy, shift_xyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_about_z(turn_rad)
    line = np.array([*line_xy, 0.0])
    T[:3, 3] = line - T[:3, :3] @ line + shift_xyz
    return T


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat / np.linalg.norm(quat))
    return R.reshape(3, 3)


class PoseVariation:
    def __init__(self, poser: Poser, data: mujoco.MjData, scene: ScenePoses, table: TableTop, cfg: dict,
                 allowed_overlap_m: float):
        self._poser, self._data, self._scene, self._table, self._cfg = poser, data, scene, table, cfg
        self._allowed_overlap_m = allowed_overlap_m
        self._joint_ranges = {hand: poser.joint_ranges_rad(hand) for hand in scene.hand_bodies}
        self._wrist_joints = set(scene.wrist_joints.values())
        self._way_to_draw_chosen_once = {
            "interaction": self._one_hand_written_pose_varied, "random": lambda rng: self._everything_at_random,
            "sparse": self._one_way_of_leaving_something_out}

    def draw(self, rng: np.random.Generator, kind: str) -> VariedPose:
        draw_again = self._way_to_draw_chosen_once[kind](rng)
        for _ in range(self._cfg["tries"]):
            varied = draw_again(rng)
            if not self._every_resting_object_is_on_the_table(varied.pose):
                continue
            self._poser.place(self._data, varied.pose)
            if (not self._poser.overlaps(self._data, self._allowed_overlap_m)
                    and not self._poser.overlaps_of_a_thing_with_itself(self._data, self._cfg["allowed_self_overlap_m"])):
                return varied
        raise RuntimeError(f"no {kind} pose" + (f" made from '{varied.made_from}'" if varied.made_from else "")
                           + f" without overlaps in {self._cfg['tries']} draws")

    def _one_hand_written_pose_varied(self, rng):
        written = self._scene.poses[rng.integers(len(self._scene.poses))]
        mirrored = bool(rng.random() < self._cfg["mirror_chance"])
        return partial(self._hand_written_pose_varied, written=written, mirrored=mirrored)

    def _hand_written_pose_varied(self, rng, written: Pose, mirrored: bool) -> VariedPose:
        pose = mirrored_left_to_right(written) if mirrored else written
        group = pose.together or (*pose.objects, *pose.hands)
        places = [pose.objects[n].xy for n in group if n in pose.objects]
        places += [pose.hands[n].T_world_palm[:2, 3] for n in group if n in pose.hands]
        centre = np.mean(places, axis=0)
        turn = rng.uniform(-np.pi, np.pi)
        to = self._table.random_point_xy(rng, self._cfg["table_margin_m"])
        lift = self._change_of_lift_of_a_held_object(rng, pose, group)
        T_moved = T_turning_about_a_vertical_line_then_shifting(turn, centre, [*(to - centre), lift])

        objects, hands = {}, {}
        for name, o in pose.objects.items():
            if name not in group:
                objects[name] = self._object_resting_at_random(rng)
                continue
            xy = (T_moved @ [*o.xy, 0.0, 1.0])[:2]
            z = None if o.z_or_none_for_resting is None else o.z_or_none_for_resting + lift
            objects[name] = ObjectPose(xy, z, o.yaw_rad + turn, o.roll_pitch_rad)
        for name, h in pose.hands.items():
            if name not in group:
                hands[name] = self._hand_at_random(rng, name)
                continue
            angles = dict(h.joint_angles_rad)
            if not self._wrist_joints & angles.keys():
                angles.update(self._wrist_at_random(rng, name))
            hands[name] = HandPose(T_moved @ h.T_world_palm, angles)
        return VariedPose(Pose(written.name, objects, hands), "interaction", written.name, mirrored)

    def _change_of_lift_of_a_held_object(self, rng, pose: Pose, group) -> float:
        for name in group:
            o = pose.objects.get(name)
            if o is not None and o.z_or_none_for_resting is not None:
                lift_now = o.z_or_none_for_resting - self._poser.resting_z(name)
                return rng.uniform(*self._cfg["held_cube_lift_m"]) - lift_now
        return 0.0

    def _everything_at_random(self, rng) -> VariedPose:
        pose = Pose("random", {name: self._object_resting_at_random(rng) for name in self._scene.object_bodies},
                    {name: self._hand_at_random(rng, name) for name in self._scene.hand_bodies})
        return VariedPose(pose, "random", "", False)

    def _one_way_of_leaving_something_out(self, rng):
        what = chosen_by_share(rng, self._cfg["sparse"])
        hand_in_view = str(rng.choice(list(self._scene.hand_bodies))) if what == "one_hand_only" else None
        return partial(self._something_left_out, what=what, hand_in_view=hand_in_view)

    def _something_left_out(self, rng, what: str, hand_in_view: str | None) -> VariedPose:
        objects = {name: self._object_resting_at_random(rng) if what == "cube_only" else self._object_out_of_view()
                   for name in self._scene.object_bodies}
        hands = {name: self._hand_at_random(rng, name) if name == hand_in_view else self._hand_out_of_view(index)
                 for index, name in enumerate(self._scene.hand_bodies)}
        return VariedPose(Pose(what, objects, hands), "sparse", what, False)

    def _object_resting_at_random(self, rng) -> ObjectPose:
        return ObjectPose(self._table.random_point_xy(rng, self._cfg["table_margin_m"]), None,
                          rng.uniform(-np.pi, np.pi))

    def _object_out_of_view(self) -> ObjectPose:
        return ObjectPose(np.zeros(2), self._cfg["out_of_view_z_m"], 0.0)

    def _hand_at_random(self, rng, hand: str) -> HandPose:
        cfg = self._cfg["random_hand"]
        T = np.eye(4)
        T[:3, :3] = random_rotation(rng)
        T[:2, 3] = self._table.random_point_xy(rng, self._cfg["table_margin_m"])
        T[2, 3] = self._table.z + rng.uniform(*cfg["palm_above_table_m"])
        ranges = self._joint_ranges[hand]
        angles = {}
        for joints in cfg["curls_together"].values():
            curl = rng.random()
            for joint in joints:
                angles[joint] = curl * ranges[joint][1]
        for joint in cfg["spread"]:
            angles[joint] = cfg["spread_fraction"] * rng.uniform(*ranges[joint])
        for joint, low_high in ranges.items():
            if joint not in angles:
                angles[joint] = rng.uniform(*low_high)
        return HandPose(T, angles)

    def _hand_out_of_view(self, index: int) -> HandPose:
        beside_the_others = [(index + 1) * SPACE_BETWEEN_THINGS_OUT_OF_VIEW_M, 0.0, self._cfg["out_of_view_z_m"]]
        return HandPose(T_world_palm(beside_the_others, [0, 1, 0], [0, 0, -1]), {})

    def _wrist_at_random(self, rng, hand: str) -> dict[str, float]:
        return {joint: rng.uniform(*self._joint_ranges[hand][joint]) for joint in sorted(self._wrist_joints)}

    def _every_resting_object_is_on_the_table(self, pose: Pose) -> bool:
        return all(self._table.is_under(o.xy) for o in pose.objects.values() if o.z_or_none_for_resting is None)
