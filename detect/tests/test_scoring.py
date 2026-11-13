import pytest

from detect.scoring import ON_A_BOX_THAT_DOES_NOT_COUNT, RIGHT, WRONG, Detection, Tally, Truth, iou, judged

HAND_BOX, CUBE_BOX, ELSEWHERE = (100, 100, 200, 200), (300, 100, 350, 150), (600, 300, 700, 400)


def test_iou_of_known_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0, "boxes that only share an edge do not overlap"


def test_perfect_detector_finds_everything_and_is_always_right():
    truths = [Truth("hand", HAND_BOX), Truth("cube", CUBE_BOX)]
    outcome = judged(truths, [Detection("cube", CUBE_BOX, 0.9), Detection("hand", HAND_BOX, 0.8)], 0.5)
    assert outcome.truth_was_found == (True, True)
    assert outcome.detection_was == (RIGHT, RIGHT)


def test_box_that_does_not_count_changes_nothing_whether_detected_or_not():
    truths = [Truth("hand", HAND_BOX), Truth("cube", CUBE_BOX, counts=False)]
    not_detected = judged(truths, [Detection("hand", HAND_BOX, 0.9)], 0.5)
    detected = judged(truths, [Detection("hand", HAND_BOX, 0.9), Detection("cube", CUBE_BOX, 0.9)], 0.5)
    assert not_detected.detection_was == (RIGHT,)
    assert detected.detection_was == (RIGHT, ON_A_BOX_THAT_DOES_NOT_COUNT)
    assert not_detected.truth_was_found == detected.truth_was_found == (True, False)


def test_two_detections_on_one_box_count_one_right_and_one_wrong_and_the_most_confident_is_the_right_one():
    outcome = judged([Truth("hand", HAND_BOX)],
                     [Detection("hand", HAND_BOX, 0.4), Detection("hand", HAND_BOX, 0.9)], 0.5)
    assert outcome.detection_was == (WRONG, RIGHT)
    assert outcome.truth_was_found == (True,)


def test_detection_of_the_wrong_class_is_wrong_and_finds_nothing():
    outcome = judged([Truth("hand", HAND_BOX)], [Detection("cube", HAND_BOX, 0.9)], 0.5)
    assert outcome.detection_was == (WRONG,)
    assert outcome.truth_was_found == (False,)


def test_detection_must_overlap_by_at_least_the_given_share():
    just_under_half = (100, 100, 200, 149)
    exactly_half = (100, 100, 200, 150)
    assert judged([Truth("hand", HAND_BOX)], [Detection("hand", just_under_half, 0.9)], 0.5).detection_was == (WRONG,)
    assert judged([Truth("hand", HAND_BOX)], [Detection("hand", exactly_half, 0.9)], 0.5).detection_was == (RIGHT,)


def test_detection_on_nothing_is_wrong():
    assert judged([], [Detection("hand", ELSEWHERE, 0.9)], 0.5).detection_was == (WRONG,)


def test_detection_between_two_boxes_is_given_the_one_it_overlaps_most():
    left, right = Truth("hand", (0, 0, 100, 100)), Truth("hand", (30, 0, 130, 100))
    overlapping_both_by_more_than_half = (25, 0, 125, 100)
    for truths, found in (([left, right], (False, True)), ([right, left], (True, False))):
        assert judged(truths, [Detection("hand", overlapping_both_by_more_than_half, 0.9)], 0.5).truth_was_found == found


def test_second_detection_takes_the_other_box_when_the_first_took_the_best_one():
    left, right = Truth("hand", (0, 0, 100, 100)), Truth("hand", (40, 0, 140, 100))
    between = (20, 0, 120, 100)
    outcome = judged([left, right], [Detection("hand", between, 0.9), Detection("hand", between, 0.8)], 0.5)
    assert outcome.detection_was == (RIGHT, RIGHT)
    assert outcome.truth_was_found == (True, True)


def test_extra_detection_on_a_found_box_is_wrong_even_beside_a_box_that_does_not_count():
    truths = [Truth("hand", HAND_BOX), Truth("hand", ELSEWHERE, counts=False)]
    outcome = judged(truths, [Detection("hand", HAND_BOX, 0.9), Detection("hand", HAND_BOX, 0.8)], 0.5)
    assert outcome.detection_was == (RIGHT, WRONG)


def test_shares_of_a_tally_and_none_when_there_is_nothing_to_divide_by():
    assert Tally(true_boxes=4, found=3, detections=5, right=3).found_share == 0.75
    assert Tally(true_boxes=4, found=3, detections=5, right=3).right_share == 0.6
    assert Tally().found_share is None and Tally().right_share is None
