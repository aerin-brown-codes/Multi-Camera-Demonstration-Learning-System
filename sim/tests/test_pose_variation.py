from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

import sim
from sim.pose_variation import PoseVariation, mirrored_left_to_right
from sim.poses import (HandPose, ObjectPose, Pose, Poser, R_world_object_turned_about_z_after_pitch_after_roll,
                       T_of_body, T_world_palm, load_scene_poses)
from sim.rig import Rig
from sim.world import build_model, top_of_box_shaped_table

CONFIGS = Path(sim.__file__).parent / "configs"
DATASET = yaml.safe_load((CONFIGS / "dataset.yaml").read_text())
CFG = DATASET["poses"]
ALLOWED_OVERLAP_M = yaml.safe_load((CONFIGS / "sim.yaml").read_text())["allowed_overlap_m"]
SAME_PLACE_WITHIN = 1e-9
FIRST_AND_MIDDLE_FINGERS_CROSSED = {"FFJ4": 0.349, "MFJ4": -0.349}
MIRROR = np.diag([-1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def cube_hands():
    model = build_model("cube_hands", Rig(0.0, []))
    scene = load_scene_poses(CONFIGS / "scenes" / "cube_hands.yaml")
    return SimpleNamespace(
        model=model, data=mujoco.MjData(model), scene=scene, poser=Poser(model, scene),
        table=top_of_box_shaped_table(model, DATASET["table_geom"]), poses={p.name: p for p in scene.poses})


def variation(s, poses, **changes):
    return PoseVariation(s.poser, s.data, replace(s.scene, poses=poses), s.table, {**CFG, **changes},
                         ALLOWED_OVERLAP_M)


def T_world_cube(s):
    return T_of_body(s.data, s.model.body(s.scene.object_bodies["cube"]).id)


def T_world_palm_of(s, hand):
    return T_of_body(s.data, s.model.body(s.scene.hand_bodies[hand].palm_body).id)


def T_world_forearm_of(s, hand):
    return T_of_body(s.data, s.model.body(s.scene.hand_bodies[hand].mocap_body).id)


def T_a_b(T_world_a, T_world_b):
    return np.linalg.inv(T_world_a) @ T_world_b


def placed(s, pose):
    s.poser.place(s.data, pose)
    return SimpleNamespace(cube=T_world_cube(s), right=T_world_palm_of(s, "right"), left=T_world_palm_of(s, "left"),
                           right_forearm=T_world_forearm_of(s, "right"), left_forearm=T_world_forearm_of(s, "left"))


def test_things_moved_together_keep_their_places_to_each_other_and_the_rest_is_placed_at_random(cube_hands):
    s = cube_hands
    held = replace(s.poses["cube_lifted"], together=("cube", "left"))
    written = placed(s, held)
    maker, rng = variation(s, [held], mirror_chance=0.0), np.random.default_rng(0)
    lifts, turns, right_palms = [], [], []
    for _ in range(40):
        varied = maker.draw(rng, "interaction")
        assert (varied.kind, varied.made_from, varied.mirrored) == ("interaction", "cube_lifted", False)
        now = placed(s, varied.pose)
        assert np.allclose(T_a_b(now.left, now.cube), T_a_b(written.left, written.cube), atol=SAME_PLACE_WITHIN), (
            "the cube must stay exactly where it was in the hand that holds it")
        assert np.allclose(now.cube[2, :3], [0, 0, 1], atol=SAME_PLACE_WITHIN), "a group only turns about the vertical"
        lifts.append(now.cube[2, 3] - s.poser.resting_z("cube"))
        turns.append(np.arctan2(now.cube[1, 0], now.cube[0, 0]))
        right_palms.append(T_a_b(now.cube, now.right))
        assert s.table.is_under(now.right[:2, 3], CFG["table_margin_m"]), "a hand outside the group is over the table"
    low, high = CFG["held_cube_lift_m"]
    assert low <= min(lifts) and max(lifts) <= high and max(lifts) - min(lifts) > 0.5 * (high - low)
    assert min(turns) < -2.0 and max(turns) > 2.0, "the group is turned every way round"
    assert not np.allclose(right_palms[0][:3, :3], right_palms[1][:3, :3], atol=0.01), (
        "the hand left out of the group is placed afresh, not carried along")


def test_pose_with_no_group_moves_as_one_piece_and_its_resting_cube_stays_resting_on_the_table(cube_hands):
    s = cube_hands
    cube_far_from_the_middle = replace(s.poses["hands_apart"], objects={"cube": ObjectPose(np.array([0.0, 0.3]), None, 0.0)})
    written = placed(s, cube_far_from_the_middle)
    maker, rng = variation(s, [cube_far_from_the_middle], mirror_chance=0.0), np.random.default_rng(1)
    centres = []
    for _ in range(40):
        varied = maker.draw(rng, "interaction")
        assert varied.pose.objects["cube"].z_or_none_for_resting is None
        now = placed(s, varied.pose)
        for hand in ("right", "left"):
            assert np.allclose(T_a_b(now.cube, getattr(now, hand)), T_a_b(written.cube, getattr(written, hand)),
                               atol=SAME_PLACE_WITHIN), "with no `together`, the whole pose moves as one piece"
        assert s.table.is_under(now.cube[:2, 3]), "a draw that would put a resting cube past the table edge is dropped"
        centres.append(np.mean([now.cube[:2, 3], now.right[:2, 3], now.left[:2, 3]], axis=0))
    assert all(s.table.is_under(c, CFG["table_margin_m"] - SAME_PLACE_WITHIN) for c in centres)
    assert np.ptp(centres, axis=0).min() > 0.3, "the pose lands all over the table"


def test_mirrored_pose_is_the_mirror_image_with_the_hands_swapped(cube_hands):
    s = cube_hands
    lifted = s.poses["cube_lifted"]
    tilted = replace(lifted, together=("cube", "left"),
                     objects={"cube": replace(lifted.objects["cube"], yaw_rad=0.3, roll_pitch_rad=(0.2, 0.4))})
    mirrored = mirrored_left_to_right(tilted)
    assert mirrored.together == ("cube", "right") and mirrored.name == tilted.name
    s.poser.place(s.data, tilted)
    written_cube, bodies = T_world_cube(s), {s.model.body(b).name: s.data.xpos[b].copy() for b in range(s.model.nbody)}
    s.poser.place(s.data, mirrored)
    assert np.allclose(T_world_cube(s)[:3, 3], MIRROR @ written_cube[:3, 3])
    assert np.allclose(T_world_cube(s)[:3, :3], MIRROR @ written_cube[:3, :3] @ MIRROR), (
        "a cube mirrored and looked at in a mirrored frame is turned the other way about z and y")
    compared = 0
    for name, written_at in bodies.items():
        if name.startswith("right/rh_"):
            twin = s.data.body("left/lh_" + name.removeprefix("right/rh_")).xpos
            assert np.allclose(twin, MIRROR @ written_at, atol=1e-6), f"{name} is not mirrored onto the left hand"
            compared += 1
    assert compared > 20
    again = mirrored_left_to_right(mirrored)
    assert again.together == tilted.together
    assert np.allclose(again.hands["left"].T_world_palm, tilted.hands["left"].T_world_palm)
    assert np.allclose(again.objects["cube"].xy, tilted.objects["cube"].xy)


def test_about_half_the_hand_written_poses_are_mirrored(cube_hands):
    maker, rng = variation(cube_hands, [cube_hands.poses["hands_apart"]]), np.random.default_rng(2)
    mirrored = sum(maker.draw(rng, "interaction").mirrored for _ in range(200))
    assert abs(mirrored / 200 - CFG["mirror_chance"]) < 0.12


def test_wrist_a_pose_leaves_open_is_drawn_and_a_written_wrist_is_kept(cube_hands):
    s = cube_hands
    apart = s.poses["hands_apart"]
    bend, side = s.scene.wrist_joints["bend_deg"], s.scene.wrist_joints["side_deg"]
    left_wrist_written = replace(apart, hands={
        "right": apart.hands["right"],
        "left": replace(apart.hands["left"], joint_angles_rad={**apart.hands["left"].joint_angles_rad, bend: -0.3})})
    maker, rng = variation(s, [left_wrist_written], mirror_chance=0.0), np.random.default_rng(3)
    ranges = s.poser.joint_ranges_rad("right")
    drawn_bends, forearm_in_palm = [], []
    for _ in range(40):
        varied = maker.draw(rng, "interaction")
        right, left = varied.pose.hands["right"].joint_angles_rad, varied.pose.hands["left"].joint_angles_rad
        assert left[bend] == -0.3 and side not in left, "a hand whose wrist the pose sets keeps it as written"
        assert ranges[bend][0] <= right[bend] <= ranges[bend][1] and ranges[side][0] <= right[side] <= ranges[side][1]
        drawn_bends.append(right[bend])
        now = placed(s, varied.pose)
        assert np.allclose(now.right, varied.pose.hands["right"].T_world_palm, atol=1e-6), (
            "bending the wrist moves the forearm, never the palm")
        forearm_in_palm.append(T_a_b(now.right, now.right_forearm)[:3, 3])
    assert np.ptp(drawn_bends) > 0.5 * (ranges[bend][1] - ranges[bend][0])
    assert np.ptp(forearm_in_palm, axis=0).max() > 0.01, "the forearm really swings about the palm"


def test_random_scene_places_the_cube_and_each_hand_on_its_own_with_nothing_overlapping(cube_hands):
    s = cube_hands
    maker, rng = variation(s, []), np.random.default_rng(4)
    low, high = CFG["random_hand"]["palm_above_table_m"]
    limits = s.poser.joint_ranges_rad("right")
    palm_faces, curls = [], []
    for _ in range(60):
        varied = maker.draw(rng, "random")
        assert (varied.kind, varied.made_from, varied.mirrored) == ("random", "", False)
        cube = varied.pose.objects["cube"]
        assert cube.z_or_none_for_resting is None and s.table.is_under(cube.xy, CFG["table_margin_m"])
        now = placed(s, varied.pose)
        assert not s.poser.overlaps(s.data, ALLOWED_OVERLAP_M)
        assert not s.poser.overlaps_of_a_thing_with_itself(s.data, CFG["allowed_self_overlap_m"])
        for hand in ("right", "left"):
            palm = getattr(now, hand)
            assert s.table.is_under(palm[:2, 3], CFG["table_margin_m"])
            assert low <= palm[2, 3] - s.table.z <= high
            palm_faces.append(-palm[:3, 1])
        angles = varied.pose.hands["right"].joint_angles_rad
        fractions = [angles[j] / limits[j][1] for j in CFG["random_hand"]["curls_together"]["first"]]
        assert np.allclose(fractions, fractions[0]) and 0.0 <= fractions[0] <= 1.0, (
            "the joints of one finger curl by the same share of their range, like a real finger")
        curls.append(fractions[0])
    palm_faces = np.array(palm_faces)
    assert (palm_faces.max(axis=0) > 0.7).all() and (palm_faces.min(axis=0) < -0.7).all(), "palms face every way"
    assert min(curls) < 0.2 and max(curls) > 0.8, "fingers go from straight to fully curled"


def test_sparse_scene_leaves_things_out_far_above_every_camera(cube_hands):
    s = cube_hands
    maker, rng = variation(s, []), np.random.default_rng(5)
    in_view_by_variant = {"empty": set(), "cube_only": {"cube"}, "one_hand_only": None}
    variants, lone_hands = Counter(), Counter()
    for _ in range(300):
        varied = maker.draw(rng, "sparse")
        assert varied.kind == "sparse"
        variants[varied.made_from] += 1
        now = placed(s, varied.pose)
        in_view = {name for name in ("cube", "right", "left")
                   if getattr(now, name)[2, 3] < s.table.z + 1.0}
        out_of_view = {"cube", "right", "left"} - in_view
        assert all(getattr(now, name)[2, 3] >= CFG["out_of_view_z_m"] for name in out_of_view)
        if varied.made_from == "one_hand_only":
            assert len(in_view) == 1 and in_view < {"right", "left"}
            lone_hands.update(in_view)
        else:
            assert in_view == in_view_by_variant[varied.made_from]
    for variant, share in CFG["sparse"].items():
        assert abs(variants[variant] / 300 - share) < 0.09, f"{variant}: {variants[variant]} of 300"
    assert min(lone_hands["right"], lone_hands["left"]) > 30, "either hand can be the one in view"


def test_hand_whose_fingers_sink_into_each_other_is_never_handed_out(cube_hands):
    s = cube_hands
    apart = s.poses["hands_apart"]
    crossed = replace(apart, hands={
        "left": apart.hands["left"],
        "right": HandPose(T_world_palm([0.3, 0.0, 1.0], [0, 1, 0], [0, 0, -1]), FIRST_AND_MIDDLE_FINGERS_CROSSED)})
    s.poser.place(s.data, crossed)
    assert not s.poser.overlaps(s.data, ALLOWED_OVERLAP_M), "the check between different things does not see this"
    with pytest.raises(RuntimeError, match="no interaction pose made from 'hands_apart' without overlaps in 5 draws"):
        variation(s, [crossed], tries=5).draw(np.random.default_rng(6), "interaction")
    deeper_than_the_fingers_cross = 0.05
    variation(s, [crossed], tries=5, allowed_self_overlap_m=deeper_than_the_fingers_cross).draw(
        np.random.default_rng(6), "interaction")


def test_pose_with_one_hand_inside_the_other_gives_up_naming_the_pose(cube_hands):
    s = cube_hands
    apart = s.poses["hands_apart"]
    both_hands_in_one_place = replace(apart, hands={"left": apart.hands["left"], "right": apart.hands["left"]})
    with pytest.raises(RuntimeError, match="no interaction pose made from 'hands_apart' without overlaps in 4 draws"):
        variation(s, [both_hands_in_one_place], tries=4).draw(np.random.default_rng(7), "interaction")


def test_what_a_scene_is_made_from_is_chosen_once_so_poses_that_often_overlap_are_not_drawn_less(cube_hands):
    s = cube_hands
    easy = replace(s.poses["hands_apart"], name="easy")
    hard = replace(s.poses["hands_apart"], name="hard", together=("cube",))
    maker, rng = variation(s, [easy, hard], mirror_chance=0.0), np.random.default_rng(8)
    made_from = Counter(maker.draw(rng, "interaction").made_from for _ in range(300))
    assert abs(made_from["hard"] / 300 - 0.5) < 0.09, (
        f"'hard' places both hands at random, which is redrawn far more often, yet must still be half: {made_from}")


def test_every_hand_written_pose_can_be_drawn_both_ways_round(cube_hands):
    s = cube_hands
    for pose in s.scene.poses:
        for mirror_chance in (0.0, 1.0):
            varied = variation(s, [pose], mirror_chance=mirror_chance).draw(np.random.default_rng(9), "interaction")
            assert varied.made_from == pose.name and varied.mirrored == bool(mirror_chance)
