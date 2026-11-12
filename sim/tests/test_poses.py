import warnings
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

import sim
from sim.poses import (HandPose, ObjectPose, Pose, PoseOverlap, Poser, T_of_body, T_world_palm, held_pose_index,
                       load_scene_poses, R_world_palm_with_fingers_along_z_and_palm_facing_minus_y)
from sim.rig import load_rig
from sim.run import frame_advancer, make_truth
from sim.world import build_model

CONFIGS = Path(sim.__file__).parent / "configs"
SIM = yaml.safe_load((CONFIGS / "sim.yaml").read_text())
ALLOWED_OVERLAP_M = SIM["allowed_overlap_m"]
SCENE_FILE = CONFIGS / "scenes" / "cube_hands.yaml"
ALLOWED_SELF_OVERLAP_M = yaml.safe_load((CONFIGS / "dataset.yaml").read_text())["poses"]["allowed_self_overlap_m"]
TABLE_TOP_Z = 0.75
TOUCHING_WITHIN_M = 0.001
MIRRORED_WITHIN_M = 1e-6
PALM_DOWN = [0, 0, -1]
AWAY_FROM_PERSON = [0, 1, 0]


@pytest.fixture(scope="module")
def cube_hands():
    rig = load_rig(CONFIGS / "rig.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = build_model("cube_hands", rig)
    scene = load_scene_poses(CONFIGS / "scenes" / "cube_hands.yaml")
    return SimpleNamespace(rig=rig, model=model, data=mujoco.MjData(model), scene=scene, poser=Poser(model, scene),
                           poses={p.name: p for p in scene.poses})


def hand(palm_pos, fingers_point=AWAY_FROM_PERSON, palm_faces=PALM_DOWN, joints=None):
    return HandPose(T_world_palm(palm_pos, fingers_point, palm_faces), joints or {})


def pose(right, left, cube=(0.0, 0.0)):
    return Pose("test", {"cube": ObjectPose(np.array(cube, float), None, 0.0)}, {"right": right, "left": left})


RIGHT_OUT_OF_THE_WAY = hand([0.45, -0.25, 0.95])
LEFT_OUT_OF_THE_WAY = hand([-0.45, -0.25, 0.95])


def body_pos(s, name):
    return s.data.body(name).xpos.copy()


def closest_m(s, owner_a, owner_b):
    m = s.model
    owners = [m.body(m.body_rootid[m.geom_bodyid[g]]).name or m.geom(g).name for g in range(m.ngeom)]
    solid = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    return min(
        mujoco.mj_geomDistance(m, s.data, a, b, 1.0, None)
        for a in solid if owners[a] == owner_a
        for b in solid if owners[b] == owner_b
    )


def test_scene_has_cube_and_two_script_moved_hands_under_the_rig_cameras(cube_hands):
    m = cube_hands.model
    for name in ["cube", "hand_right", "hand_left"]:
        assert m.body(name).id > 0
    assert m.nmocap == 2 and m.body("hand_right").mocapid[0] >= 0 and m.body("hand_left").mocapid[0] >= 0
    assert [m.camera(c.name).name for c in cube_hands.rig.cameras] == [c.name for c in cube_hands.rig.cameras]
    assert "board" not in [m.body(i).name for i in range(m.nbody)], "the board belongs to the calibration scene"
    bodies = make_truth(m, cube_hands.data, cube_hands.rig, 0)["bodies"]
    assert {"cube", "right/rh_palm", "left/lh_ffdistal"} <= bodies.keys()


@pytest.mark.parametrize("side,prefix,thumb_side_sign", [("right", "right/rh_", 1), ("left", "left/lh_", -1)])
def test_palm_lands_where_asked_and_faces_the_asked_way(cube_hands, side, prefix, thumb_side_sign):
    s = cube_hands
    palm_pos = np.array([0.1, -0.05, 1.0])
    fingers_point = np.array([0.3, 0.8, -0.2]) / np.linalg.norm([0.3, 0.8, -0.2])
    palm_faces = np.cross(fingers_point, [1.0, 0.2, 0.0])
    palm_faces /= np.linalg.norm(palm_faces)
    wrist_bent = {"WRJ1": -0.4, "WRJ2": 0.1}

    def placed(joints):
        tested = hand(palm_pos, fingers_point, palm_faces, joints)
        other = LEFT_OUT_OF_THE_WAY if side == "right" else RIGHT_OUT_OF_THE_WAY
        s.poser.place(s.data, Pose("test", {"cube": ObjectPose(np.zeros(2), None, 0.0)}, {side: tested, **{
            ("left" if side == "right" else "right"): other}}))

    placed(wrist_bent)
    palm = body_pos(s, prefix + "palm")
    assert np.allclose(palm, palm_pos, atol=1e-9), "a bent wrist must move the forearm, not the palm"
    to_knuckle = body_pos(s, prefix + "mfknuckle") - palm
    assert to_knuckle @ fingers_point / np.linalg.norm(to_knuckle) > 0.99
    thumb_side = thumb_side_sign * np.cross(fingers_point, palm_faces)
    assert (body_pos(s, prefix + "thbase") - palm) @ thumb_side > 0.02, f"{side} thumb is on the wrong side"

    fingertip_open = body_pos(s, prefix + "mfdistal")
    placed({**wrist_bent, "MFJ3": 1.0})
    curl = body_pos(s, prefix + "mfdistal") - fingertip_open
    assert curl @ palm_faces > 0.03, "fingers must curl towards the side the palm faces"


def test_palm_faces_is_squared_up_against_fingers_point():
    R = R_world_palm_with_fingers_along_z_and_palm_facing_minus_y([0, 1, 0], [0, 0.5, -1])
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R[:, 2], [0, 1, 0]) and np.allclose(R[:, 1], [0, 0, 1])
    with pytest.raises(ValueError, match="parallel"):
        R_world_palm_with_fingers_along_z_and_palm_facing_minus_y([0, 1, 0], [0, -2, 0])


def test_same_finger_shape_gives_the_mirror_image_on_the_left_hand(cube_hands):
    s = cube_hands
    shape = s.poses["cube_lifted"].hands["left"].joint_angles_rad
    assert any(shape.values())
    s.poser.place(s.data, pose(hand([0.2, 0.0, 0.95], joints=shape), hand([-0.2, 0.0, 0.95], joints=shape)))
    m = s.model
    right_bodies = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("right/rh_")]
    assert len(right_bodies) > 20
    for name in right_bodies:
        mirrored = body_pos(s, name) * [-1, 1, 1]
        left = body_pos(s, name.replace("right/rh_", "left/lh_"))
        assert np.allclose(left, mirrored, atol=MIRRORED_WITHIN_M), name


def test_placing_a_pose_leaves_nothing_behind_from_the_one_before(cube_hands):
    s = cube_hands

    def state_after(*names):
        for name in names:
            s.poser.place(s.data, s.poses[name])
        return np.concatenate([s.data.qpos, s.data.mocap_pos.ravel(), s.data.mocap_quat.ravel()])

    assert np.array_equal(state_after("hands_apart"), state_after("hand_around_cube", "cube_lifted", "hands_apart"))


def test_every_pose_in_the_file_passes_the_overlap_check(cube_hands):
    assert len(cube_hands.scene.poses) >= 7
    cube_hands.poser.require_no_overlaps(cube_hands.data, cube_hands.scene.poses, ALLOWED_OVERLAP_M)


@pytest.mark.parametrize("name,owner_a,owner_b", [
    ("hand_beside_cube", "hand_right", "cube"),
    ("hand_around_cube", "hand_right", "cube"),
    ("cube_lifted", "hand_left", "cube"),
    ("hands_touching", "hand_right", "hand_left"),
    ("hand_flat_on_cube", "hand_left", "cube"),
    ("fingertip_on_cube", "hand_right", "cube"),
    ("cube_pinched_on_table", "hand_right", "cube"),
    ("cube_wrapped_hand_hovering", "hand_left", "cube"),
    ("cube_gripped_from_side", "hand_right", "cube"),
    ("cube_lifted_tilted", "hand_left", "cube"),
    ("cube_on_palm", "hand_left", "cube"),
    ("cube_held_palm_up", "hand_left", "cube"),
    ("cube_in_fist", "hand_right", "cube"),
    ("cube_in_fist", "hand_right", "hand_left"),
    ("cube_between_palms", "hand_right", "cube"),
    ("cube_between_palms", "hand_left", "cube"),
    ("cube_passed", "hand_right", "cube"),
    ("cube_passed", "hand_left", "cube"),
    ("fingertips_touching", "hand_right", "hand_left"),
    ("hand_resting_on_hand", "hand_right", "hand_left"),
    ("wrists_crossed", "hand_right", "hand_left"),
])
def test_poses_meant_to_touch_do_touch(cube_hands, name, owner_a, owner_b):
    cube_hands.poser.place(cube_hands.data, cube_hands.poses[name])
    assert closest_m(cube_hands, owner_a, owner_b) < TOUCHING_WITHIN_M


@pytest.mark.parametrize("name,owner_a,owner_b", [
    ("hands_cupped_around_cube", "hand_right", "cube"),
    ("hands_cupped_around_cube", "hand_left", "cube"),
    ("hands_cupped_around_cube", "hand_right", "hand_left"),
    ("cube_wrapped_hand_hovering", "hand_right", "hand_left"),
    ("cube_wrapped_hand_hovering", "hand_right", "cube"),
    ("both_fists", "hand_right", "hand_left"),
])
def test_poses_meant_to_stay_apart_do_not_touch(cube_hands, name, owner_a, owner_b):
    cube_hands.poser.place(cube_hands.data, cube_hands.poses[name])
    assert closest_m(cube_hands, owner_a, owner_b) > TOUCHING_WITHIN_M


def test_hovering_and_crossing_poses_hide_without_touching(cube_hands):
    s = cube_hands
    s.poser.place(s.data, s.poses["hand_above_cube"])
    assert closest_m(s, "hand_right", "cube") > 0.02
    assert np.allclose(body_pos(s, "right/rh_mfknuckle")[:2], body_pos(s, "cube")[:2], atol=0.03)
    s.poser.place(s.data, s.poses["hands_overlapping"])
    assert closest_m(s, "hand_right", "hand_left") > TOUCHING_WITHIN_M
    over = body_pos(s, "left/lh_mfknuckle") - body_pos(s, "right/rh_mfknuckle")
    assert np.linalg.norm(over[:2]) < 0.05 and over[2] > 0.03, "the left hand must be above the right one"
    s.poser.place(s.data, s.poses["cube_wrapped_hand_hovering"])
    over = body_pos(s, "right/rh_mfknuckle") - body_pos(s, "left/lh_mfknuckle")
    assert np.linalg.norm(over[:2]) < 0.05 and over[2] > 0.03, "the right hand must be above the left one"
    s.poser.place(s.data, s.poses["hands_cupped_around_cube"])
    assert max(closest_m(s, "hand_right", "cube"), closest_m(s, "hand_left", "cube")) < 0.03, "cupped means close"


@pytest.mark.parametrize("right,left,owners,parts", [
    (hand([0.0, -0.10, 0.785]), LEFT_OUT_OF_THE_WAY, {"hand_right", "cube"}, None),
    (hand([0.25, -0.10, 0.76]), LEFT_OUT_OF_THE_WAY, {"hand_right", "table"}, None),
    (RIGHT_OUT_OF_THE_WAY, hand([-0.25, -0.10, 0.76]), {"hand_left", "table"}, None),
    (hand([0.02, 0.2, 0.95]), hand([-0.02, 0.2, 0.95]), {"hand_right", "hand_left"}, None),
    (hand([0.25, 0.2, 0.79], fingers_point=[0, 1, 0.4]), LEFT_OUT_OF_THE_WAY,
     {"hand_right", "table"}, {"right/rh_forearm", "table"}),
    (hand([0.0, 0.25, 0.95]), hand([0.2, 0.05, 0.95], fingers_point=[1, 0, 0]),
     {"hand_right", "hand_left"}, {"right/rh_forearm", "left/lh_forearm"}),
], ids=["fingers_in_cube", "right_hand_in_table", "left_hand_in_table", "hand_in_hand",
        "forearm_in_table", "forearm_in_forearm"])
def test_pose_pushed_into_something_is_rejected_naming_the_pair(cube_hands, right, left, owners, parts):
    s = cube_hands
    bad = pose(right, left, cube=(0.0, 0.0))
    s.poser.place(s.data, bad)
    found = {frozenset(o.owners): o for o in s.poser.overlaps(s.data, ALLOWED_OVERLAP_M)}
    assert set(found) == {frozenset(owners)}
    assert not s.poser.overlaps_of_a_thing_with_itself(s.data, ALLOWED_SELF_OVERLAP_M), (
        "two things sunk into each other are not one thing sunk into itself")
    if parts:
        assert set(found[frozenset(owners)].parts) == parts, "MuJoCo reports no contact between shapes that cannot move"
    with pytest.raises(PoseOverlap, match="pose 'test'.*" + ".*".join(found[frozenset(owners)].owners)):
        s.poser.require_no_overlaps(s.data, [bad], ALLOWED_OVERLAP_M)


def test_shapes_that_only_touch_are_within_the_allowance(cube_hands):
    s = cube_hands
    s.poser.place(s.data, pose(RIGHT_OUT_OF_THE_WAY, LEFT_OUT_OF_THE_WAY))
    assert s.poser.overlaps(s.data, ALLOWED_OVERLAP_M) == [], "the cube resting on the table is not an overlap"


def test_pose_must_place_every_hand_and_object(cube_hands):
    only_right = Pose("half", {"cube": ObjectPose(np.zeros(2), None, 0.0)}, {"right": RIGHT_OUT_OF_THE_WAY})
    with pytest.raises(ValueError, match="pose 'half' must place exactly"):
        cube_hands.poser.place(cube_hands.data, only_right)


@pytest.mark.parametrize("joints,complaint", [
    ({"FFJ3": 3.0}, "outside its range"),
    ({"FFJ9": 0.1}, "no joints"),
])
def test_impossible_finger_angles_are_refused(cube_hands, joints, complaint):
    with pytest.raises(ValueError, match=complaint):
        cube_hands.poser.place(cube_hands.data, pose(hand([0.3, 0, 0.95], joints=joints), LEFT_OUT_OF_THE_WAY))


def test_unknown_finger_shape_is_refused_when_the_file_loads(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "scenes" / "cube_hands.yaml").read_text())
    cfg["poses"][0]["right"]["fingers"] = "shape_nobody_wrote"
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="no finger shape named 'shape_nobody_wrote'"):
        load_scene_poses(path)


def test_each_pose_is_held_then_the_list_loops():
    index = [held_pose_index(frame, fps=30, hold_s=3.0, pose_count=7) for frame in (0, 89, 90, 629, 630)]
    assert index == [0, 0, 1, 6, 0]


def test_scene_with_poses_is_placed_and_never_simulated(cube_hands):
    s = cube_hands
    data = mujoco.MjData(s.model)
    advance_to, mode = frame_advancer(SIM, CONFIGS, "cube_hands", s.model, data, s.rig.fps)
    assert "no physics" in mode
    lifted = [p.name for p in s.scene.poses].index("cube_lifted")
    first_frame = round(lifted * SIM["pose_hold_s"] * s.rig.fps)
    heights = []
    for frame in range(first_frame, first_frame + 10):
        advance_to(frame)
        assert np.isclose(data.time, frame / s.rig.fps)
        heights.append(data.body("cube").xpos[2])
    assert heights[0] > TABLE_TOP_Z + 0.1 and len(set(heights)) == 1, "a lifted cube must not fall"


def test_scene_without_poses_still_runs_physics():
    rig = load_rig(CONFIGS / "rig.yaml")
    model = build_model("tabletop", rig)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start_z = data.body("brick_red").xpos[2]
    advance_to, mode = frame_advancer(SIM, CONFIGS, "tabletop", model, data, rig.fps)
    assert mode == "physics"
    advance_to(30)
    assert 1.0 <= data.time < 1.0 + 2 * model.opt.timestep
    assert data.body("brick_red").xpos[2] < start_z - 0.001, "bricks start 2 mm up and must settle"


def scene_with_one_pose(tmp_path, **changes_to_the_pose):
    cfg = yaml.safe_load(SCENE_FILE.read_text())
    high_above_the_table = {"fingers_point": AWAY_FROM_PERSON, "palm_faces": PALM_DOWN, "fingers": "open"}
    cfg["poses"] = [{"name": "test", "cube": {"pos": [0.0, 0.2, 1.0]},
                     "right": {"palm_pos": [0.3, 0.0, 1.0], **high_above_the_table},
                     "left": {"palm_pos": [-0.3, 0.0, 1.0], **high_above_the_table}}]
    for name, change in changes_to_the_pose.items():
        if name == "together":
            cfg["poses"][0]["together"] = change
        else:
            cfg["poses"][0][name].update(change)
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def forearm_and_palm(s, pose_to_place):
    s.poser.place(s.data, pose_to_place)
    return {side: SimpleNamespace(elbow_end=s.data.geom(f"{side}/{side[0]}h_forearm").xpos.copy(),
                                  palm=T_of_body(s.data, s.model.body(f"{side}/{side[0]}h_palm").id))
            for side in ("right", "left")}


def test_wrist_setting_moves_the_forearm_and_leaves_the_palm_where_it_was(cube_hands, tmp_path):
    s = cube_hands
    straight = forearm_and_palm(s, load_scene_poses(scene_with_one_pose(tmp_path)).poses[0])
    raised = forearm_and_palm(s, load_scene_poses(scene_with_one_pose(
        tmp_path, right={"wrist": {"bend_deg": -30}}, left={"wrist": {"bend_deg": -30}})).poses[0])
    swung = forearm_and_palm(s, load_scene_poses(scene_with_one_pose(
        tmp_path, right={"wrist": {"side_deg": -20}}, left={"wrist": {"side_deg": -20}})).poses[0])
    for side, away_from_the_thumb in (("right", 1.0), ("left", -1.0)):
        for bent in (raised, swung):
            assert np.allclose(bent[side].palm, straight[side].palm, atol=1e-6), "the palm must not move"
        lifted = raised[side].elbow_end - straight[side].elbow_end
        assert lifted[2] > 0.05 and abs(lifted[0]) < 1e-6, (
            f"bend_deg below 0 raises the forearm of a palm-down hand, here by {lifted[2] * 100:.1f} cm")
        sideways = swung[side].elbow_end - straight[side].elbow_end
        assert away_from_the_thumb * sideways[0] > 0.03 and abs(sideways[2]) < 1e-6, (
            "side_deg below 0 swings the forearm away from the thumb's side, mirrored on the left hand")
    assert np.allclose(raised["left"].elbow_end, raised["right"].elbow_end * [-1, 1, 1], atol=MIRRORED_WITHIN_M)


def test_wrist_limits_are_the_hand_files_own(cube_hands):
    for hand_name in ("right", "left"):
        ranges = cube_hands.poser.joint_ranges_rad(hand_name)
        assert np.allclose(np.rad2deg(ranges["WRJ1"]), [-40, 28], atol=0.5), "bend_deg, as the scene file says"
        assert np.allclose(np.rad2deg(ranges["WRJ2"]), [-30, 10], atol=0.5), "side_deg, as the scene file says"
        assert "FFJ3" in ranges and len(ranges) > 20


@pytest.mark.parametrize("changes,complaint", [
    ({"right": {"wrist": {"twist_deg": 10}}}, r"wrist has no setting 'twist_deg', only \['bend_deg', 'side_deg'\]"),
    ({"together": ["cube", "middle"]}, r"together names \['middle'\], which the scene does not have"),
])
def test_unknown_wrist_setting_or_group_member_is_refused_when_the_file_loads(tmp_path, changes, complaint):
    with pytest.raises(ValueError, match=complaint):
        load_scene_poses(scene_with_one_pose(tmp_path, **changes))


def test_wrist_set_by_both_the_finger_shape_and_the_pose_is_refused(tmp_path):
    path = scene_with_one_pose(tmp_path, right={"wrist": {"bend_deg": 10}, "fingers": "bent_wrist"})
    cfg = yaml.safe_load(path.read_text())
    cfg["finger_shapes"]["bent_wrist"] = {"WRJ1": 0.2}
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="finger shape 'bent_wrist' already sets the wrist joint WRJ1"):
        load_scene_poses(path)


def test_group_and_tilt_are_read_from_the_file(tmp_path):
    loaded = load_scene_poses(scene_with_one_pose(
        tmp_path, together=["cube", "left"], cube={"tilt_deg": [10, -20], "yaw_deg": 30})).poses[0]
    assert loaded.together == ("cube", "left")
    assert np.allclose(np.rad2deg(loaded.objects["cube"].roll_pitch_rad), [10, -20])
    assert load_scene_poses(scene_with_one_pose(tmp_path)).poses[0].together == ()


@pytest.mark.parametrize("yaw_deg,roll_deg,pitch_deg,cube_axis,points_along", [
    (0, 90, 0, "y", [0, 0, 1]),
    (0, 0, 90, "x", [0, 0, -1]),
    (90, 0, 90, "y", [-1, 0, 0]),
    (0, 90, 90, "z", [0, -1, 0]),
    (90, 90, 90, "z", [1, 0, 0]),
])
def test_cube_is_rolled_about_its_x_then_pitched_about_y_then_turned_about_the_vertical(
        cube_hands, yaw_deg, roll_deg, pitch_deg, cube_axis, points_along):
    s = cube_hands
    tilted = ObjectPose(np.array([0.0, 0.2]), 1.0, np.deg2rad(yaw_deg), tuple(np.deg2rad([roll_deg, pitch_deg])))
    s.poser.place(s.data, Pose("test", {"cube": tilted}, {"right": RIGHT_OUT_OF_THE_WAY, "left": LEFT_OUT_OF_THE_WAY}))
    R_world_cube = s.data.body("cube").xmat.reshape(3, 3)
    assert np.allclose(R_world_cube[:, "xyz".index(cube_axis)], points_along, atol=1e-9)
    assert np.allclose(s.data.body("cube").xpos, [0.0, 0.2, 1.0])


def test_fingers_sunk_into_each_other_are_found_only_by_the_check_of_a_thing_with_itself(cube_hands):
    s = cube_hands
    first_and_middle_fingers_crossed = {"FFJ4": 0.349, "MFJ4": -0.349}
    s.poser.place(s.data, pose(hand([0.3, 0, 1.0], joints=first_and_middle_fingers_crossed), LEFT_OUT_OF_THE_WAY))
    assert not s.poser.overlaps(s.data, ALLOWED_OVERLAP_M), "one hand is one thing to the check between things"
    (crossed,) = s.poser.overlaps_of_a_thing_with_itself(s.data, ALLOWED_SELF_OVERLAP_M)
    assert crossed.owners == ("hand_right", "hand_right") and crossed.depth_m > 0.01
    assert set(crossed.parts) == {"right/rh_ffproximal", "right/rh_mfproximal"}
    assert not s.poser.overlaps_of_a_thing_with_itself(s.data, crossed.depth_m), "deeper than it is: nothing"


def test_no_pose_in_the_file_has_a_hand_sunk_into_itself(cube_hands):
    s = cube_hands
    for p in s.scene.poses:
        s.poser.place(s.data, p)
        sunk = s.poser.overlaps_of_a_thing_with_itself(s.data, ALLOWED_SELF_OVERLAP_M)
        assert not sunk, f"pose '{p.name}': " + "; ".join(map(str, sunk)) + " (the dataset would never use it)"
