import warnings
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

import sim
from sim.poses import (HandPose, ObjectPose, Pose, PoseOverlap, Poser, T_world_palm, held_pose_index,
                       load_scene_poses, R_world_palm_with_fingers_along_z_and_palm_facing_minus_y)
from sim.rig import load_rig
from sim.run import frame_advancer, make_truth
from sim.world import build_model

CONFIGS = Path(sim.__file__).parent / "configs"
SIM = yaml.safe_load((CONFIGS / "sim.yaml").read_text())
ALLOWED_OVERLAP_M = SIM["allowed_overlap_m"]
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
])
def test_poses_meant_to_touch_do_touch(cube_hands, name, owner_a, owner_b):
    cube_hands.poser.place(cube_hands.data, cube_hands.poses[name])
    assert closest_m(cube_hands, owner_a, owner_b) < TOUCHING_WITHIN_M


def test_hovering_and_crossing_poses_hide_without_touching(cube_hands):
    s = cube_hands
    s.poser.place(s.data, s.poses["hand_above_cube"])
    assert closest_m(s, "hand_right", "cube") > 0.02
    assert np.allclose(body_pos(s, "right/rh_mfknuckle")[:2], body_pos(s, "cube")[:2], atol=0.03)
    s.poser.place(s.data, s.poses["hands_overlapping"])
    assert closest_m(s, "hand_right", "hand_left") > TOUCHING_WITHIN_M
    over = body_pos(s, "left/lh_mfknuckle") - body_pos(s, "right/rh_mfknuckle")
    assert np.linalg.norm(over[:2]) < 0.05 and over[2] > 0.03, "the left hand must be above the right one"


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
    cfg["poses"][0]["right"]["fingers"] = "fist"
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="no finger shape named 'fist'"):
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
