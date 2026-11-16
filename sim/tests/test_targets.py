import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

import sim
from sim.poses import HandPose, ObjectPose, Pose, Poser, T_world_palm, load_scene_poses
from sim.render import RigRenderer
from sim.rig import load_rig
from sim.run import make_truth, target_boxes_or_none_without_targets
from sim.targets import NO_TARGET, Target, body_and_every_body_below_it, load_targets, target_of_each_geom
from sim.world import build_model

CONFIGS = Path(sim.__file__).parent / "configs"
SIM = yaml.safe_load((CONFIGS / "sim.yaml").read_text())
SCENE_CONFIG = CONFIGS / "scenes" / "cube_hands.yaml"
CUBE_HALF_SIDE_M = 0.025
CUBE_RESTING_Z = 0.775
INSIDE_THE_TABLE_Z = 0.4
FAR_ABOVE_EVERY_CAMERA_Z = 10.0
BOX_MATCHES_CAMERA_MATHS_WITHIN_PX = 1.0


@pytest.fixture(scope="module")
def cube_hands():
    rig = load_rig(CONFIGS / "rig.yaml")
    model = build_model("cube_hands", rig)
    scene = load_scene_poses(SCENE_CONFIG)
    renderer = RigRenderer(model, rig)
    targets = load_targets(SCENE_CONFIG)
    yield SimpleNamespace(
        rig=rig, model=model, data=mujoco.MjData(model), poser=Poser(model, scene), renderer=renderer,
        poses={p.name: p for p in scene.poses}, targets=targets, geom_groups_when_loaded=model.geom_group.copy(),
        target_boxes=target_boxes_or_none_without_targets(SIM, CONFIGS, "cube_hands", model, renderer))
    renderer.close()


def hand_out_of_view(x):
    return HandPose(T_world_palm([x, 0.0, FAR_ABOVE_EVERY_CAMERA_Z], [0, 1, 0], [0, 0, -1]), {})


def only_the_cube_in_view(cube: ObjectPose) -> Pose:
    return Pose("test", {"cube": cube}, {"right": hand_out_of_view(0.3), "left": hand_out_of_view(-0.3)})


def boxes_of(s, pose: Pose) -> dict:
    s.poser.place(s.data, pose)
    return s.target_boxes.in_every_camera(s.data)


def pixel_of(cam, point_world) -> np.ndarray:
    in_cam = cam.R_world_cam.T @ (np.asarray(point_world) - cam.position)
    return (cam.profile.K @ (in_cam / in_cam[2]))[:2]


def is_inside(inner_box, outer_box) -> bool:
    return (inner_box[0] >= outer_box[0] and inner_box[1] >= outer_box[1]
            and inner_box[2] <= outer_box[2] and inner_box[3] <= outer_box[3])


def test_targets_are_the_cube_and_each_hand_from_the_wrist_on(cube_hands):
    m, targets = cube_hands.model, cube_hands.targets
    assert [(t.name, t.class_name) for t in targets] == [
        ("cube", "cube"), ("hand_right", "hand"), ("hand_left", "hand")]
    target_of_geom = target_of_each_geom(m, targets)

    def targets_of_body(body):
        return {int(target_of_geom[g]) for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id}

    assert targets_of_body("cube") == {0}
    for index, prefix in [(1, "right/rh_"), (2, "left/lh_")]:
        for part in ["wrist", "palm", "ffdistal", "lfmetacarpal", "thdistal"]:
            assert targets_of_body(prefix + part) == {index}, f"{prefix}{part} belongs to the hand"
        assert targets_of_body(prefix + "forearm") == {NO_TARGET}, "the forearm is not part of the hand box"
    assert target_of_geom[m.geom("table").id] == NO_TARGET and target_of_geom[m.geom("floor").id] == NO_TARGET


def test_body_listed_under_two_targets_is_refused(cube_hands):
    hand_and_its_own_fingertip = [Target("hand", "hand", "right/rh_wrist"), Target("tip", "hand", "right/rh_ffdistal")]
    with pytest.raises(ValueError, match="in two targets: 'hand' and 'tip'"):
        target_of_each_geom(cube_hands.model, hand_and_its_own_fingertip)


def test_cube_alone_gets_the_box_around_its_projected_corners_in_every_camera(cube_hands):
    s = cube_hands
    xy, yaw = np.array([0.08, -0.05]), np.deg2rad(20)
    boxes = boxes_of(s, only_the_cube_in_view(ObjectPose(xy, None, yaw)))
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    corners = [np.array([*xy, CUBE_RESTING_Z]) + R @ (CUBE_HALF_SIDE_M * np.array([sx, sy, sz]))
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    for cam in s.rig.cameras:
        assert boxes[cam.name].keys() == {"cube"}, "hands far outside the view are left out, not listed as hidden"
        cube = boxes[cam.name]["cube"]
        corner_pixels = np.array([pixel_of(cam, c) for c in corners])
        pixel_centre_to_pixel_edge = 0.5
        expected = np.concatenate([corner_pixels.min(axis=0), corner_pixels.max(axis=0)]) + pixel_centre_to_pixel_edge
        error_px = np.abs(np.array(cube["box"]) - expected).max()
        assert error_px <= BOX_MATCHES_CAMERA_MATHS_WITHIN_PX, (
            f"{cam.name}: box {cube['box']} is {error_px:.2f} px from the projected corners {expected.round(2)}")
        assert cube["visible_fraction"] == 1.0 and cube["counts"] and cube["class"] == "cube"


def test_hand_held_over_the_cube_hides_part_of_it(cube_hands):
    s = cube_hands
    hidden = s.poses["hand_above_cube"]
    with_hand = boxes_of(s, hidden)
    without_hand = boxes_of(s, only_the_cube_in_view(hidden.objects["cube"]))
    cameras_with_the_cube_partly_hidden = 0
    for cam in s.rig.cameras:
        covered, clear = with_hand[cam.name]["cube"], without_hand[cam.name]["cube"]
        assert is_inside(covered["box"], clear["box"]), f"{cam.name}: {covered['box']} outside {clear['box']}"
        assert covered["visible_px"] == round(covered["visible_fraction"] * clear["visible_px"])
        if covered["visible_fraction"] < 1.0:
            cameras_with_the_cube_partly_hidden += 1
            assert covered["visible_px"] < clear["visible_px"]
    assert cameras_with_the_cube_partly_hidden >= 1


def test_cube_inside_the_table_is_in_view_but_fully_hidden(cube_hands):
    s = cube_hands
    boxes = boxes_of(s, only_the_cube_in_view(ObjectPose(np.zeros(2), INSIDE_THE_TABLE_Z, 0.0)))
    for cam in s.rig.cameras:
        assert boxes[cam.name] == {"cube": {"class": "cube", "box": None, "visible_fraction": 0.0, "visible_px": 0,
                                            "counts": False}}


def test_hand_box_stops_before_the_elbow(cube_hands):
    s, m = cube_hands, cube_hands.model
    boxes = boxes_of(s, s.poses["hands_apart"])
    elbows_in_the_image = 0
    for target, prefix in [("hand_right", "right/rh_"), ("hand_left", "left/lh_")]:
        forearm = m.geom(prefix + "forearm").id
        centre, along = s.data.geom_xpos[forearm], s.data.geom_xmat[forearm].reshape(3, 3)[:, 2]
        ends = [centre + sign * m.geom_size[forearm][1] * along for sign in (-1, 1)]
        wrist = s.data.body(prefix + "wrist").xpos
        elbow = max(ends, key=lambda end: np.linalg.norm(end - wrist))
        for cam in s.rig.cameras:
            u, v = pixel_of(cam, elbow)
            x0, y0, x1, y1 = boxes[cam.name][target]["box"]
            assert not (x0 <= u < x1 and y0 <= v < y1), f"{cam.name}: {target} box reaches the elbow"
            elbows_in_the_image += 0 <= u < cam.profile.width and 0 <= v < cam.profile.height
    assert elbows_in_the_image >= 4, "the elbows must be in the picture for this to mean anything"


def test_every_pose_gives_boxes_inside_the_image_that_fit_in_a_truth_message(cube_hands):
    s = cube_hands
    for pose in s.poses.values():
        s.poser.place(s.data, pose)
        truth = json.loads(json.dumps(make_truth(s.model, s.data, s.rig, 0, s.target_boxes)))
        assert truth["boxes"].keys() == {cam.name for cam in s.rig.cameras}
        for cam in s.rig.cameras:
            for name, b in truth["boxes"][cam.name].items():
                assert 0.0 <= b["visible_fraction"] <= 1.0, (
                    f"{pose.name} {cam.name} {name}: more pixels visible than when drawn alone")
                if b["box"] is not None:
                    x0, y0, x1, y1 = b["box"]
                    assert 0 <= x0 < x1 <= cam.profile.width and 0 <= y0 < y1 <= cam.profile.height


def test_colour_and_depth_are_the_same_before_and_after_the_box_draws(cube_hands):
    s = cube_hands
    s.poser.place(s.data, s.poses["hands_overlapping"])
    before = s.renderer.render(s.data)
    s.target_boxes.in_every_camera(s.data)
    after = s.renderer.render(s.data)
    assert np.array_equal(s.model.geom_group, s.geom_groups_when_loaded), "hidden shapes must be shown again"
    for cam in s.rig.cameras:
        assert np.array_equal(before[cam.name][0], after[cam.name][0]), f"{cam.name}: colour changed"
        assert np.array_equal(before[cam.name][1], after[cam.name][1]), f"{cam.name}: depth changed"


def test_failed_draw_of_one_target_alone_leaves_every_shape_drawn(cube_hands):
    s = cube_hands
    s.poser.place(s.data, s.poses["hands_apart"])
    cam = s.rig.cameras[0]
    before = s.renderer.render(s.data)
    only_the_cube = np.arange(s.model.ngeom) == s.model.geom("cube").id
    with pytest.raises(Exception):
        s.renderer.geom_at_each_pixel(None, cam, drawing_only=only_the_cube)
    assert np.array_equal(s.model.geom_group, s.geom_groups_when_loaded), "hidden shapes must be shown again"
    assert np.array_equal(s.renderer.render(s.data)[cam.name][0], before[cam.name][0])


def test_truth_carries_a_point_for_each_target_the_centre_of_mass_of_its_bodies(cube_hands):
    s = cube_hands
    s.poser.place(s.data, s.poses["cube_lifted"])
    truth = json.loads(json.dumps(make_truth(s.model, s.data, s.rig, 0, s.target_boxes)))
    assert truth["target_points"].keys() == truth["boxes"]["cam0"].keys() == {t.name for t in s.targets}
    assert truth["target_points"]["cube"] == pytest.approx(truth["bodies"]["cube"]["pos"], abs=1e-12), \
        "the cube's point is the cube body's position"
    for target in s.targets:
        bodies = body_and_every_body_below_it(s.model, s.model.body(target.body).id)
        masses = s.model.body_mass[bodies]
        centre_of_mass = masses @ s.data.xipos[bodies] / masses.sum()
        assert truth["target_points"][target.name] == pytest.approx(centre_of_mass, abs=1e-9), target.name
    wrist = truth["bodies"]["right/rh_wrist"]["pos"]
    assert np.linalg.norm(np.subtract(truth["target_points"]["hand_right"], wrist)) > 0.04, \
        "a hand's point is inside the palm, not at the wrist where its top body starts"


def test_scene_without_targets_sends_no_boxes(tabletop):
    renderer = RigRenderer(tabletop.model, tabletop.rig)
    target_boxes = target_boxes_or_none_without_targets(SIM, CONFIGS, "tabletop", tabletop.model, renderer)
    renderer.close()
    assert target_boxes is None
    truth = make_truth(tabletop.model, tabletop.data, tabletop.rig, 0, target_boxes)
    assert "boxes" not in truth and "target_points" not in truth
