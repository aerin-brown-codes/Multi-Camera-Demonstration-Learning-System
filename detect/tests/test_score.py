import json
import shutil

import pytest
import yaml

from detect.score import BY_CLASS, BY_VISIBLE_FRACTION, as_markdown, meets_the_bar, score_run, scores_of, tallies_by_split
from detect.scoring import Detection, Tally, Truth
from detect.truth import CAMERA_DISTANCE, SCENE_KIND, ImageWithTruth
from conftest import box_of_scene, write_image_and_labels, write_scene

HAND_BOX, CUBE_BOX, HIDDEN_BOX, ELSEWHERE = [100, 100, 200, 200], [300, 100, 350, 150], [500, 50, 560, 110], (600, 300, 700, 400)


@pytest.fixture
def test_sets(cfg, tmp_path):
    write_scene(tmp_path / "sim", "test", 0, "interaction", {
        "cam0": (0.5, 80.0, {"hand": box_of_scene("hand", HAND_BOX, 0.9), "cube": box_of_scene("cube", CUBE_BOX, 0.3),
                             "hidden": box_of_scene("hand", HIDDEN_BOX, 0.05, counts=False)}),
        "cam1": (1.2, 20.0, {"cube": box_of_scene("cube", CUBE_BOX, 0.5)})})
    write_scene(tmp_path / "sim", "test", 1, "sparse", {"cam0": (0.5, 80.0, {})})
    write_image_and_labels(tmp_path / "real", "test", "frame", [(0, tuple(HAND_BOX))], suffix=".jpg")
    return cfg


def detector_that_reports(detections_by_image_name: dict):
    return lambda images: (detections_by_image_name.get(image.name, []) for image in images)


PERFECT = {"000000_cam0.png": [Detection("hand", tuple(HAND_BOX), 0.9), Detection("cube", tuple(CUBE_BOX), 0.9)],
           "000000_cam1.png": [Detection("cube", tuple(CUBE_BOX), 0.9)],
           "frame.jpg": [Detection("hand", tuple(HAND_BOX), 0.9)]}


def test_perfect_detector_scores_100_percent_everywhere_and_meets_the_bar(test_sets):
    scores = scores_of(test_sets, detector_that_reports(PERFECT))
    assert scores["meets_the_bar"]
    for scored in scores["sets"].values():
        for groups in scored["splits"].values():
            for row in groups.values():
                assert row["found_share"] in (1.0, None) and row["right_share"] in (1.0, None)
    assert scores["sets"]["sim"]["images"] == 3 and scores["sets"]["real"]["images"] == 1
    assert scores["sets"]["sim"]["splits"][BY_CLASS]["all"]["true_boxes"] == 3, "the hidden hand is not to be found"


def test_detecting_the_box_that_does_not_count_leaves_every_number_as_it_was(test_sets):
    also_hidden = {**PERFECT, "000000_cam0.png": [*PERFECT["000000_cam0.png"], Detection("hand", tuple(HIDDEN_BOX), 0.9)]}
    assert scores_of(test_sets, detector_that_reports(also_hidden)) == scores_of(test_sets, detector_that_reports(PERFECT))


def test_sim_results_are_split_by_class_hidden_share_kind_of_scene_and_camera(test_sets):
    missing_the_hidden_cube = {**PERFECT, "000000_cam0.png": [Detection("hand", tuple(HAND_BOX), 0.9)]}
    splits = scores_of(test_sets, detector_that_reports(missing_the_hidden_cube))["sets"]["sim"]["splits"]
    assert splits[BY_CLASS]["cube"]["found_share"] == 0.5 and splits[BY_CLASS]["hand"]["found_share"] == 1.0
    assert splits[BY_VISIBLE_FRACTION]["cube, 0.1 to 0.4"]["found"] == 0
    assert splits[BY_VISIBLE_FRACTION]["cube, 0.4 to 0.7"]["found"] == 1
    assert splits[BY_VISIBLE_FRACTION]["hand, 0.7 to 1"]["found"] == 1
    assert splits[SCENE_KIND]["interaction"]["found_share"] == pytest.approx(2 / 3)
    assert splits[CAMERA_DISTANCE]["0.4 to 0.7"]["true_boxes"] == 2 and splits[CAMERA_DISTANCE]["1 to 1.5"]["found"] == 1


def test_real_results_are_split_by_class_only(test_sets):
    assert list(scores_of(test_sets, detector_that_reports(PERFECT))["sets"]["real"]["splits"]) == [BY_CLASS]


def test_false_detection_on_an_empty_image_is_charged_to_its_class_and_its_kind_of_scene(test_sets):
    seeing_things = {**PERFECT, "000001_cam0.png": [Detection("cube", ELSEWHERE, 0.9)]}
    splits = scores_of(test_sets, detector_that_reports(seeing_things))["sets"]["sim"]["splits"]
    assert splits[BY_CLASS]["cube"]["right_share"] == pytest.approx(2 / 3)
    assert splits[BY_CLASS]["hand"]["right_share"] == 1.0
    assert splits[SCENE_KIND]["sparse"] == {"true_boxes": 0, "found": 0, "detections": 1, "right": 0,
                                            "found_share": None, "right_share": 0.0}


def test_bar_is_missed_when_one_class_of_one_set_is_under_it(test_sets):
    scores = scores_of(test_sets, detector_that_reports({**PERFECT, "000000_cam1.png": []}))
    assert not scores["sets"]["sim"]["meets_the_bar"] and scores["sets"]["real"]["meets_the_bar"]
    assert not scores["meets_the_bar"]


def test_bar_looks_at_all_classes_together_and_each_class_that_has_something_to_find():
    bar = {"found": 0.9, "right": 0.9}
    good, poor = Tally(10, 10, 10, 10), Tally(10, 8, 8, 8)
    assert meets_the_bar({BY_CLASS: {"all": good, "hand": good}}, bar)
    assert not meets_the_bar({BY_CLASS: {"all": good, "hand": good, "cube": poor}}, bar)
    assert not meets_the_bar({BY_CLASS: {"all": Tally(10, 10, 12, 10), "hand": good}}, bar), "too many false detections"
    assert meets_the_bar({BY_CLASS: {"all": good, "hand": good, "cube": Tally(0, 0, 0, 0)}}, bar)
    assert not meets_the_bar({BY_CLASS: {"all": Tally(10, 0, 0, 0)}}, bar), "nothing detected at all"
    assert not meets_the_bar({}, bar), "a set with nothing to find proves nothing"


def test_detections_must_come_one_list_per_image(cfg):
    image = ImageWithTruth("a.png", (Truth("hand", tuple(HAND_BOX)),), {})
    with pytest.raises(ValueError):
        tallies_by_split([image, image], [[]], cfg["score"])


def test_scoring_a_run_reads_its_own_config_and_writes_the_table_beside_the_weights(test_sets, tmp_path):
    run_dir = tmp_path / "runs" / "first"
    run_dir.mkdir(parents=True)
    (run_dir / "train.yaml").write_text(yaml.safe_dump(test_sets))
    scores = score_run(run_dir, detector_that_reports(PERFECT))
    assert json.loads((run_dir / "scores.json").read_text()) == scores
    table = (run_dir / "scores.md").read_text()
    assert table == as_markdown(scores)
    assert "## sim: 3 images, bar met" in table
    assert "| class | cube | 2 | 2 | 100.0% | 2 | 2 | 100.0% |" in table
    assert "| visible fraction | cube, 0.1 to 0.4 | 1 | 1 | 100.0% | 0 | 0 | - |" in table
