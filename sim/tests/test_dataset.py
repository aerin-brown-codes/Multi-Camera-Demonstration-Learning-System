import json
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
import pytest

from sim.dataset import (DEFAULT_CONFIG, SceneMaker, label_line, label_lines, scene_file, scenes_in_set,
                         what_is_in_a_set, write_set)
from sim.poses import HandPose, ObjectPose, Pose, Poser, T_world_palm, load_scene_poses
from sim.render import RigRenderer
from sim.rig import Rig, mujoco_quat
from sim.scene_variation import ExtraObject, apply_look, name_of_extra, random_look
from sim.world import build_model, top_of_box_shaped_table

SCENES_WRITTEN = {"train": 8, "check": 3, "test": 3}
ROUNDING_OF_A_LABEL = 1e-6
SCENE_CONFIG = DEFAULT_CONFIG.parent / "scenes" / "cube_hands.yaml"


@pytest.fixture(scope="module")
def maker():
    return SceneMaker(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def written(maker, tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    for set_name, scenes in SCENES_WRITTEN.items():
        assert write_set(maker, out, set_name, scenes) == scenes
    descriptions = {(set_name, number): json.loads(scene_file(out, set_name, number).read_text())
                    for set_name, scenes in SCENES_WRITTEN.items() for number in range(scenes)}
    return SimpleNamespace(out=out, descriptions=descriptions)


def files_of(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_label_line_is_class_then_centre_then_size_as_shares_of_the_image():
    assert label_line(1, [100, 80, 160, 130], 848, 480) == "1 0.153302 0.218750 0.070755 0.104167"
    assert label_line(0, [0, 0, 848, 480], 848, 480) == "0 0.500000 0.500000 1.000000 1.000000"


def test_only_a_box_that_counts_gets_a_label_line():
    boxes = {"cube": {"class": "cube", "box": [100, 80, 160, 130], "counts": True},
             "right": {"class": "hand", "box": [0, 0, 5, 5], "counts": False},
             "left": {"class": "hand", "box": [0, 0, 848, 480], "counts": True}}
    assert label_lines(boxes, {"hand": 0, "cube": 1}, 848, 480) == [
        "1 0.153302 0.218750 0.070755 0.104167", "0 0.500000 0.500000 1.000000 1.000000"]


def test_sets_are_whole_scenes_adding_up_to_the_images_asked_for(maker):
    assert scenes_in_set(5000, 4) == 1250 and scenes_in_set(500, 4) == 125 and scenes_in_set(1000, 4) == 250
    assert scenes_in_set(5, 4) == 2, "a part scene is rounded up, never dropped"
    seeds = [s["seed"] for s in maker.cfg["sets"].values()]
    assert len(set(seeds)) == len(seeds), "two sets with one seed would hold the same scenes"


def test_every_scene_writes_its_images_with_a_label_file_each_and_one_description(written):
    cameras = [f"cam{i}" for i in range(4)]
    for (set_name, number), d in written.descriptions.items():
        assert (d["set"], d["scene"]) == (set_name, number) and list(d["cameras"]) == cameras
        width, height = d["image_size"]
        for cam in cameras:
            stem = f"{number:06d}_{cam}"
            image = cv2.imread(str(written.out / "images" / set_name / f"{stem}.png"))
            assert image.shape == (height, width, 3)
            lines = (written.out / "labels" / set_name / f"{stem}.txt").read_text().splitlines()
            counting = [b for b in d["boxes"][cam].values() if b["counts"]]
            assert len(lines) == len(counting), f"{set_name} {stem}: one line per box that counts, no others"
            for line, box in zip(lines, counting):
                class_id, x, y, w, h = line.split()
                assert int(class_id) == {"hand": 0, "cube": 1}[box["class"]]
                x0, y0, x1, y1 = box["box"]
                assert float(x) * width == pytest.approx((x0 + x1) / 2, abs=width * ROUNDING_OF_A_LABEL)
                assert float(y) * height == pytest.approx((y0 + y1) / 2, abs=height * ROUNDING_OF_A_LABEL)
                assert float(w) * width == pytest.approx(x1 - x0, abs=width * ROUNDING_OF_A_LABEL)
                assert float(h) * height == pytest.approx(y1 - y0, abs=height * ROUNDING_OF_A_LABEL)
                assert 0.0 < float(w) <= 1.0 and 0.0 < float(h) <= 1.0
    assert len(files_of(written.out)) == sum(SCENES_WRITTEN.values()) * (4 + 4 + 1)


def test_description_keeps_what_scoring_needs(written, maker):
    cfg = maker.cfg
    table = top_of_box_shaped_table(build_model(cfg["scene"], Rig(0.0, [])), cfg["table_geom"])
    (near, far), (low, high) = cfg["cameras"]["distance_m"], cfg["cameras"]["height_deg"]
    for key, d in written.descriptions.items():
        assert d["kind"] in cfg["scene_kinds"] and d["kind"] == maker.kind_of_scene(*key)
        assert {"made_from", "mirrored", "scene_builds", "look"} <= d.keys()
        for cam in d["cameras"].values():
            T_world_cam, position = np.array(cam["T_world_cam"]), np.array(cam["position"])
            assert np.allclose(T_world_cam[:3, 3], position)
            assert near <= cam["distance_m"] <= far and low <= cam["height_deg"] <= high
            assert position[2] > table.z and table.is_under(cam["aim"][:2])
            assert np.linalg.norm(position - cam["aim"]) == pytest.approx(cam["distance_m"])
            aim_in_cam = T_world_cam[:3, :3].T @ (np.array(cam["aim"]) - position)
            assert np.allclose(aim_in_cam, [0, 0, cam["distance_m"]]), "the stored pose looks at the stored aim"
            assert 1 <= cam["draws"] <= cfg["camera_tries"]


def test_same_seed_writes_the_same_bytes_and_a_stopped_run_carries_on(written, maker, tmp_path):
    for set_name in SCENES_WRITTEN:
        write_set(maker, tmp_path, set_name, 2)
    first = files_of(written.out)
    again = files_of(tmp_path)
    assert len(again) == len(SCENES_WRITTEN) * 2 * 9
    for name, content in again.items():
        assert content == first[name], f"{name} differs between two runs"

    assert write_set(maker, tmp_path, "train", 2) == 0, "scenes already written are kept"
    scene_file(tmp_path, "train", 0).unlink()
    assert write_set(maker, tmp_path, "train", 2) == 1, "a scene with no description is written again"
    assert files_of(tmp_path) == again, "scene N depends on the seed and N alone, not on what was made before it"


def test_the_three_sets_share_no_scene(written):
    cameras_of_set = {
        set_name: {tuple(d["cameras"]["cam0"]["position"]) for key, d in written.descriptions.items()
                   if key[0] == set_name}
        for set_name in SCENES_WRITTEN}
    assert len(cameras_of_set["train"]) == SCENES_WRITTEN["train"], "scenes of one set differ from each other"
    assert not cameras_of_set["train"] & cameras_of_set["check"]
    assert not cameras_of_set["train"] & cameras_of_set["test"]
    assert not cameras_of_set["check"] & cameras_of_set["test"]


def test_scene_kinds_come_out_near_their_shares(maker):
    for set_name, set_cfg in maker.cfg["sets"].items():
        scenes = scenes_in_set(set_cfg["images"], maker.cfg["cameras_per_scene"])
        kinds = Counter(maker.kind_of_scene(set_name, n) for n in range(scenes))
        for kind, share in maker.cfg["scene_kinds"].items():
            assert abs(kinds[kind] / scenes - share) < 2.5 * np.sqrt(share * (1 - share) / scenes) + 0.01, (
                f"{set_name}: {kinds[kind]} of {scenes} scenes are {kind}, asked for {share:.0%}")


def test_every_built_scene_has_nothing_sinking_into_anything_extras_included(maker):
    scene_poses = load_scene_poses(SCENE_CONFIG)
    rng = np.random.default_rng(0)
    extras, built_again = 0, 0
    for index in range(30):
        built = maker.build_with_nothing_overlapping(rng, ["interaction", "random", "sparse"][index % 3])
        poser, data = Poser(built.model, scene_poses), mujoco.MjData(built.model)
        poser.place(data, built.varied.pose)
        assert not poser.overlaps(data, maker._allowed_overlap_m)
        assert built.model.ncam == maker.cfg["cameras_per_scene"] == len(built.cameras)
        extras += len(built.look.extras)
        built_again += built.builds > 1
    assert extras > 30 and built_again > 0, "the check must have had extras to look at and scenes to turn down"


def test_extra_that_sinks_in_is_put_somewhere_else_and_nothing_else_about_the_scene_changes(maker, monkeypatch):
    poses_drawn, looks_drawn = [], []
    draw_a_pose = maker._pose_variation.draw
    monkeypatch.setattr(maker._pose_variation, "draw",
                        lambda rng, kind: poses_drawn.append(draw_a_pose(rng, kind)) or poses_drawn[-1])
    monkeypatch.setattr("sim.dataset.random_look", lambda *a: looks_drawn.append(random_look(*a)) or looks_drawn[-1])
    what_it_is = lambda extra: (extra.shape, extra.half_sizes_as_mujoco_wants_them, extra.rgb)
    rng = np.random.default_rng(1)
    put_somewhere_else = 0
    for index in range(30):
        built = maker.build_with_nothing_overlapping(rng, ["interaction", "random", "sparse"][index % 3])
        (pose,), (first_look,) = poses_drawn, looks_drawn
        poses_drawn.clear(), looks_drawn.clear()
        assert built.varied is pose, "or a pose that extras often sink into would come up less than the others"
        assert [what_it_is(e) for e in built.look.extras] == [what_it_is(e) for e in first_look.extras], (
            "or scenes with many or large extras would come up less than the plan says")
        assert replace(built.look, extras=[]) == replace(first_look, extras=[])
        moved = sum(not np.array_equal(now.position, first.position)
                    for now, first in zip(built.look.extras, first_look.extras))
        assert (moved > 0) == (built.builds > 1)
        put_somewhere_else += moved
    assert put_somewhere_else > 0, "no extra sank into anything, so nothing was tried"


@pytest.mark.parametrize("where, sinks_into", [
    ("cube", "cube"), ("right forearm", "hand_right"), ("left palm", "hand_left"), ("other extra", name_of_extra(1))])
def test_extra_object_inside_something_is_seen_by_the_overlap_check(maker, where, sinks_into):
    cfg = maker.cfg
    scene_poses = load_scene_poses(SCENE_CONFIG)
    plain = build_model(cfg["scene"], Rig(0.0, []))
    table = top_of_box_shaped_table(plain, cfg["table_geom"])
    hand = lambda x: HandPose(T_world_palm([x, 0.0, table.z + 0.2], [0, 1, 0], [0, 0, -1]), {})
    pose = Pose("test", {"cube": ObjectPose(np.array([0.0, 0.2]), None, 0.0)}, {"right": hand(0.3), "left": hand(-0.3)})
    plain_data = mujoco.MjData(plain)
    Poser(plain, scene_poses).place(plain_data, pose)
    far_corner = np.array([-0.5, -0.3, table.z + 0.03])
    place = {"cube": plain_data.body("cube").xpos, "right forearm": plain_data.geom("right/rh_forearm").xpos,
             "left palm": plain_data.body("left/lh_palm").xpos, "other extra": far_corner}[where]
    ball = lambda position: ExtraObject("ball", (0.03,), np.array(position), np.eye(3), (0.1, 0.2, 0.9), False)
    look = replace(random_look(np.random.default_rng(0), table, np.empty((0, 2)), cfg["look"]),
                   extras=[ball(place), ball(far_corner)])
    model = build_model(cfg["scene"], Rig(0.0, []), partial(
        apply_look, look=look, table_geom=cfg["table_geom"], floor_geom=cfg["floor_geom"],
        skin_cfg=cfg["look"]["skin"], backdrop_px=cfg["look"]["backdrop_px"]))
    poser, data = Poser(model, scene_poses), mujoco.MjData(model)
    poser.place(data, pose)
    pairs = [set(o.owners) for o in poser.overlaps(data, maker._allowed_overlap_m)]
    assert {name_of_extra(0), sinks_into} in pairs, f"an extra inside the {where} went unseen: {pairs}"
    assert all(name_of_extra(1) not in pair for pair in pairs) or where == "other extra", (
        "an extra resting on the table away from everything is no overlap")


def test_image_of_a_scene_with_targets_shows_one_and_a_sparse_scene_shows_only_what_it_kept(maker, written):
    tries = maker.cfg["camera_tries"]
    for d in written.descriptions.values():
        if d["kind"] == "sparse":
            continue
        for cam, boxes in d["boxes"].items():
            assert any(b["counts"] for b in boxes.values()) or d["cameras"][cam]["draws"] == tries, (
                "a camera that sees no target is drawn again")

    kept = {"empty": set(), "cube_only": {"cube"}}
    seen = set()
    sparse_scenes = (n for n in range(10_000) if maker.kind_of_scene("train", n) == "sparse")
    while seen != {"empty", "cube_only", "one_hand_only"}:
        d = maker.scene("train", next(sparse_scenes)).description
        seen.add(d["made_from"])
        in_any_image = {name for boxes in d["boxes"].values() for name in boxes}
        if d["made_from"] == "one_hand_only":
            assert len(in_any_image) <= 1 and in_any_image <= {"hand_right", "hand_left"}
        else:
            assert in_any_image <= kept[d["made_from"]], f"{d['made_from']} scene shows {in_any_image}"
        assert all(cam["draws"] == 1 for cam in d["cameras"].values()), "an image with no target is kept as it is"


def test_camera_written_down_is_where_the_image_was_taken_from_after_a_camera_was_drawn_again(
        maker, written, monkeypatch):
    models = []
    monkeypatch.setattr("sim.dataset.RigRenderer",
                        lambda model, rig: models.append(model) or RigRenderer(model, rig))
    with_a_camera_drawn_again = [key for key, d in written.descriptions.items()
                                 if any(cam["draws"] > 1 for cam in d["cameras"].values())]
    assert with_a_camera_drawn_again, "no written scene exercises the redraw; write more scenes in this test file"
    for set_name, number in with_a_camera_drawn_again:
        description = maker.scene(set_name, number).description
        model = models.pop()
        for name, cam in description["cameras"].items():
            in_model = model.camera(name).id
            assert np.allclose(model.cam_pos[in_model], cam["position"]), (
                f"{set_name} {number} {name}: scoring by camera distance would use a camera that took no image")
            assert np.allclose(model.cam_pos[in_model], np.array(cam["T_world_cam"])[:3, 3])
            assert np.allclose(model.cam_quat[in_model], mujoco_quat(np.array(cam["T_world_cam"])[:3, :3]))


def test_no_image_is_black_or_washed_out(written):
    for path in sorted((written.out / "images").rglob("*.png")):
        brightness = cv2.imread(str(path)).mean()
        assert 15 < brightness < 235, f"{path.name} has mean brightness {brightness:.0f} of 255"


def test_summary_of_a_set_counts_what_was_written(written):
    d = [d for key, d in written.descriptions.items() if key[0] == "train"]
    counting = Counter(b["class"] for scene in d for boxes in scene["boxes"].values() for b in boxes.values()
                       if b["counts"])
    summary = what_is_in_a_set(written.out, "train")
    assert summary.startswith(f"train: {SCENES_WRITTEN['train']} scenes (")
    assert f"{4 * SCENES_WRITTEN['train']} images" in summary
    assert f"{counting['cube']} cube" in summary and f"{counting['hand']} hand" in summary
