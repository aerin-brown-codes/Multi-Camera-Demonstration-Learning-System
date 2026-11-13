import pytest

from detect.labels import label_file_of, label_line
from detect.scoring import Truth
from detect.truth import (CAMERA_DISTANCE, CAMERA_HEIGHT, SCENE_KIND, group_between_edges,
                          images_with_truth_from_labels, images_with_truth_from_scenes)
from conftest import IMAGE_SIZE, box_of_scene, write_image_and_labels, write_scene

CLASS_NAMES = ["hand", "cube"]


def test_group_between_edges_takes_the_lower_edge_in_and_the_very_last_edge_too():
    edges = [0.1, 0.4, 0.7, 1.0]
    assert group_between_edges(0.1, edges) == "0.1 to 0.4"
    assert group_between_edges(0.4, edges) == "0.4 to 0.7"
    assert group_between_edges(1.0, edges) == "0.7 to 1"
    assert group_between_edges(0.05, edges) == "below 0.1"
    assert group_between_edges(1.2, edges) == "above 1"


def test_scene_file_gives_each_image_its_boxes_and_the_groups_it_is_scored_in(cfg, tmp_path):
    write_scene(tmp_path, "test", 7, "interaction", {
        "cam0": (0.5, 80.0, {"cube": box_of_scene("cube", [10, 10, 60, 60], 0.3),
                             "hand_left": box_of_scene("hand", [100, 100, 200, 200], 0.05, counts=False),
                             "hand_right": {"class": "hand", "box": None, "visible_fraction": 0.0, "counts": False}}),
        "cam1": (1.2, 20.0, {})})
    cam0, cam1 = images_with_truth_from_scenes(tmp_path, "test", CLASS_NAMES, cfg["score"])
    assert cam0.path.name == "000007_cam0.png"
    assert cam0.truths == (Truth("cube", (10, 10, 60, 60), True, 0.3),
                           Truth("hand", (100, 100, 200, 200), False, 0.05)), "a fully hidden target has no box"
    assert cam0.groups == {SCENE_KIND: "interaction", CAMERA_DISTANCE: "0.4 to 0.7", CAMERA_HEIGHT: "70 to 90"}
    assert cam1.truths == () and cam1.groups[CAMERA_DISTANCE] == "1 to 1.5"


def test_scene_file_and_label_file_that_disagree_stop_the_scoring(cfg, tmp_path):
    write_scene(tmp_path, "test", 0, "random", {"cam0": (0.5, 80.0, {"cube": box_of_scene("cube", [10, 10, 60, 60])})})
    with pytest.raises(ValueError, match="out of step with step 3"):
        images_with_truth_from_scenes(tmp_path, "test", ["cube", "hand"], cfg["score"])


def test_scene_file_whose_image_is_gone_stops_the_scoring(cfg, tmp_path):
    write_scene(tmp_path, "test", 0, "random", {"cam0": (0.5, 80.0, {})})
    (tmp_path / "images" / "test" / "000000_cam0.png").unlink()
    with pytest.raises(FileNotFoundError, match="000000_cam0.png"):
        images_with_truth_from_scenes(tmp_path, "test", CLASS_NAMES, cfg["score"])


def test_label_file_gives_boxes_in_pixels_that_all_count(tmp_path):
    image = write_image_and_labels(tmp_path, "test", "frame", [(0, (100, 50, 300, 250))], suffix=".jpg")
    (only,) = images_with_truth_from_labels(tmp_path, "test", CLASS_NAMES)
    assert only.path == image and only.groups == {}
    (truth,) = only.truths
    assert (truth.class_name, truth.counts, truth.visible_fraction) == ("hand", True, None)
    assert truth.box == pytest.approx((100, 50, 300, 250), abs=1e-3)
