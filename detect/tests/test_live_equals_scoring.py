import cv2
import pytest

from detect.config import CONFIGS, from_repo, load_config
from detect.score import CONFIG_OF_A_RUN, CUT_OFF_OF_A_RUN, WEIGHTS_OF_A_RUN, picked_cut_off
from detect.scoring import iou

LIVE = load_config(CONFIGS / "live.yaml")
RUN_DIR = from_repo(LIVE["runs_dir"]) / LIVE["run"]
SIM_TEST_IMAGES = from_repo("datasets/sim/images/test")
IMAGES_COMPARED = 20
CAMERAS_IN_ONE_LIVE_CALL = 4
ROOM_FOR_A_BOX_SITTING_ON_THE_CUT_OFF = 0.02
OVERLAP_OF_THE_SAME_BOX = 0.95

on_this_machine = all((RUN_DIR / file).is_file() for file in (WEIGHTS_OF_A_RUN, CONFIG_OF_A_RUN, CUT_OFF_OF_A_RUN)) \
    and SIM_TEST_IMAGES.is_dir()


def has_a_partner(detection, among) -> bool:
    return any(other.class_name == detection.class_name and iou(other.box, detection.box) >= OVERLAP_OF_THE_SAME_BOX
               for other in among)


@pytest.mark.skipif(not on_this_machine, reason=f"needs the scored run {RUN_DIR} and the images in {SIM_TEST_IMAGES}")
def test_the_real_weights_find_the_same_boxes_in_arrays_from_the_feed_as_in_the_files_scoring_reads():
    from detect.detector import detector_from_weights, detector_of_a_run

    files = sorted(SIM_TEST_IMAGES.glob("*.png"))[:IMAGES_COMPARED]
    assert len(files) == IMAGES_COMPARED
    scoring = detector_from_weights(RUN_DIR / WEIGHTS_OF_A_RUN, load_config(RUN_DIR / CONFIG_OF_A_RUN))
    as_scoring_finds = list(scoring(files))

    as_the_feed_gives_them = [cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB) for file in files]
    live = detector_of_a_run(RUN_DIR, LIVE["device"])
    as_live_finds = [detections for start in range(0, IMAGES_COMPARED, CAMERAS_IN_ONE_LIVE_CALL)
                     for detections in live(as_the_feed_gives_them[start:start + CAMERAS_IN_ONE_LIVE_CALL])]

    cut_off = picked_cut_off(RUN_DIR)
    clearly_kept = cut_off + ROOM_FOR_A_BOX_SITTING_ON_THE_CUT_OFF
    assert sum(d.confidence >= clearly_kept for detections in as_live_finds for d in detections) >= IMAGES_COMPARED, \
        "live found almost nothing in images that hold a cube and hands"
    for file, scoring_found, live_found in zip(files, as_scoring_finds, as_live_finds, strict=True):
        assert all(d.confidence >= cut_off for d in live_found), f"{file.name}: live kept a box below the cut-off"
        for detection in scoring_found:
            if detection.confidence >= clearly_kept:
                assert has_a_partner(detection, live_found), f"{file.name}: scoring finds {detection}, live does not"
        for detection in live_found:
            if detection.confidence >= clearly_kept:
                assert has_a_partner(detection, scoring_found), f"{file.name}: live finds {detection}, scoring does not"
