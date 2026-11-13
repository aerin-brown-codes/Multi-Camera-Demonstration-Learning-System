import json

import cv2
import pytest
import yaml

from detect.mistakes import FALSE_BGR, MISSED_BGR, write_pictures
from detect.scoring import Detection
from conftest import box_of_scene, write_image_and_labels, write_scene

HAND_BOX, CUBE_BOX, ELSEWHERE = [100, 100, 200, 200], [300, 100, 350, 150], (600, 300, 700, 400)


@pytest.fixture
def scored_run(cfg, tmp_path):
    for scene in range(4):
        write_scene(tmp_path / "sim", "check", scene, "interaction", {
            "cam0": (0.5, 80.0, {"hand": box_of_scene("hand", HAND_BOX), "cube": box_of_scene("cube", CUBE_BOX, 0.3)})})
    write_image_and_labels(tmp_path / "real", "check", "frame", [(0, tuple(HAND_BOX))], suffix=".jpg")
    run_dir = tmp_path / "runs" / "first"
    run_dir.mkdir(parents=True)
    (run_dir / "train.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "cut_off.json").write_text(json.dumps({"picked": 0.5}))
    return run_dir


def detector_that_reports(detections_by_image_name: dict):
    return lambda images: (detections_by_image_name.get(image.name, []) for image in images)


def pictures_in(run_dir) -> list[str]:
    return sorted(str(picture.relative_to(run_dir / "mistakes")) for picture in (run_dir / "mistakes").rglob("*")
                  if picture.is_file())


FINDS_THE_HAND = [Detection("hand", tuple(HAND_BOX), 0.9)]
FINDS_BOTH = [*FINDS_THE_HAND, Detection("cube", tuple(CUBE_BOX), 0.9)]


def test_each_kind_of_mistake_gets_a_folder_and_an_image_with_nothing_wrong_gets_no_picture(scored_run):
    drawn = write_pictures(scored_run, detector_that_reports({
        "000000_cam0.png": FINDS_THE_HAND, "000001_cam0.png": FINDS_BOTH, "000002_cam0.png": FINDS_BOTH,
        "000003_cam0.png": FINDS_BOTH, "frame.jpg": [*FINDS_THE_HAND, Detection("cube", ELSEWHERE, 0.9)]}))
    assert drawn == {"sim": {"missed cube, 0.1 to 0.4 visible": (1, 1)}, "real": {"false cube": (1, 1)}}
    assert pictures_in(scored_run) == ["real/false cube/frame.jpg", "sim/missed cube, 0.1 to 0.4 visible/000000_cam0.png"]


def test_missed_box_and_false_detection_are_drawn_in_their_own_colours(scored_run):
    write_pictures(scored_run, detector_that_reports({
        "000000_cam0.png": [*FINDS_THE_HAND, Detection("hand", ELSEWHERE, 0.9)], "000001_cam0.png": FINDS_BOTH,
        "000002_cam0.png": FINDS_BOTH, "000003_cam0.png": FINDS_BOTH}))
    picture = cv2.imread(str(scored_run / "mistakes" / "sim" / "false hand" / "000000_cam0.png"))
    assert tuple(picture[CUBE_BOX[1], CUBE_BOX[0]]) == MISSED_BGR, "the cube nobody detected"
    assert tuple(picture[ELSEWHERE[1], ELSEWHERE[0]]) == FALSE_BGR, "the hand that is not there"
    assert (scored_run / "mistakes" / "sim" / "missed cube, 0.1 to 0.4 visible" / "000000_cam0.png").exists(), \
        "an image with two kinds of mistake is in both folders"


def test_no_more_pictures_of_a_kind_than_the_config_asks_for(scored_run):
    drawn = write_pictures(scored_run, detector_that_reports({}))
    assert drawn["sim"]["missed hand, 0.7 to 1 visible"] == (2, 4)
    assert len(list((scored_run / "mistakes" / "sim" / "missed hand, 0.7 to 1 visible").iterdir())) == 2


def test_detection_less_sure_than_the_picked_cut_off_is_not_a_false_detection(scored_run):
    drawn = write_pictures(scored_run, detector_that_reports({"frame.jpg": [*FINDS_THE_HAND, Detection("hand", ELSEWHERE, 0.4)]}))
    assert "real" not in drawn or drawn["real"] == {}


def test_pictures_of_an_earlier_scoring_do_not_linger(scored_run):
    write_pictures(scored_run, detector_that_reports({}))
    everything = {f"{scene:06d}_cam0.png": FINDS_BOTH for scene in range(4)} | {"frame.jpg": FINDS_THE_HAND}
    write_pictures(scored_run, detector_that_reports(everything))
    assert pictures_in(scored_run) == []
