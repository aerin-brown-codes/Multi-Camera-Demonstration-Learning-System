import numpy as np
import pytest

from sim.lift_error import box_of_each_target_that_counts, lift_error, overlap_as_intersection_over_union

CAMERA = np.array([0.5, -0.5, 1.25])
TRUTH_POINT = np.array([0.1, 0.05, 0.8])
TOWARDS_THE_CAMERA = (CAMERA - TRUTH_POINT) / np.linalg.norm(CAMERA - TRUTH_POINT)
TO_THE_SIDE = np.cross(TOWARDS_THE_CAMERA, [0.0, 0.0, 1.0]) / np.linalg.norm(np.cross(TOWARDS_THE_CAMERA, [0, 0, 1]))
EXACT_MM = 1e-9
HALF_OVERLAP = 0.5


def truth_box(class_name, x0_y0_x1_y1, counts=True):
    return {"class": class_name, "box": x0_y0_x1_y1, "visible_fraction": 1.0, "visible_px": 1, "counts": counts}


def model_box(class_name, x0_y0_x1_y1, confidence=0.9):
    return class_name, tuple(float(edge) for edge in x0_y0_x1_y1), confidence


def test_a_point_30_mm_towards_the_camera_is_30_in_front_and_not_to_the_side():
    error = lift_error(CAMERA, TRUTH_POINT, TRUTH_POINT + 0.030 * TOWARDS_THE_CAMERA)
    assert (error.sideways_mm, error.along_mm, error.distance_mm) == pytest.approx((0.0, -30.0, 30.0), abs=EXACT_MM)


def test_a_point_10_mm_to_the_side_is_10_sideways_and_neither_in_front_nor_behind():
    error = lift_error(CAMERA, TRUTH_POINT, TRUTH_POINT + 0.010 * TO_THE_SIDE)
    assert (error.sideways_mm, error.along_mm, error.distance_mm) == pytest.approx((10.0, 0.0, 10.0), abs=EXACT_MM)


def test_a_point_behind_and_to_the_side_splits_into_both_and_distance_is_the_whole():
    error = lift_error(CAMERA, TRUTH_POINT, TRUTH_POINT - 0.004 * TOWARDS_THE_CAMERA + 0.003 * TO_THE_SIDE)
    assert (error.sideways_mm, error.along_mm, error.distance_mm) == pytest.approx((3.0, 4.0, 5.0), abs=EXACT_MM)


def test_overlap_is_intersection_over_union():
    assert overlap_as_intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert overlap_as_intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert overlap_as_intersection_over_union((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_two_hands_under_one_box_give_one_paired_hand_and_one_missed():
    truth = {"hand_right": truth_box("hand", [0, 0, 100, 100]), "hand_left": truth_box("hand", [10, 0, 110, 100])}
    box_of, on_no_target = box_of_each_target_that_counts(truth, [model_box("hand", [8, 0, 108, 100])], HALF_OVERLAP)
    assert box_of == {"hand_right": None, "hand_left": 0}, "the box goes to the hand it overlaps most"
    assert on_no_target == []


def test_a_box_of_the_other_class_is_not_paired_however_well_it_fits():
    truth = {"cube": truth_box("cube", [0, 0, 100, 100])}
    box_of, on_no_target = box_of_each_target_that_counts(truth, [model_box("hand", [0, 0, 100, 100])], HALF_OVERLAP)
    assert box_of == {"cube": None} and on_no_target == [0]


def test_a_target_that_does_not_count_is_not_reported_and_a_box_on_it_is_on_no_target():
    truth = {"cube": truth_box("cube", [0, 0, 100, 100], counts=False), "hand_left": truth_box("hand", None, False)}
    box_of, on_no_target = box_of_each_target_that_counts(truth, [model_box("cube", [0, 0, 100, 100])], HALF_OVERLAP)
    assert box_of == {} and on_no_target == [0]


def test_a_box_that_overlaps_less_than_asked_is_not_paired():
    truth = {"cube": truth_box("cube", [0, 0, 100, 100])}
    just_under, exactly = model_box("cube", [0, 0, 100, 49.9]), model_box("cube", [0, 0, 100, 50])
    assert box_of_each_target_that_counts(truth, [just_under], HALF_OVERLAP) == ({"cube": None}, [0])
    assert box_of_each_target_that_counts(truth, [exactly], HALF_OVERLAP) == ({"cube": 0}, [])


def test_the_largest_overlap_is_paired_first_and_each_box_and_target_is_used_once():
    truth = {"hand_right": truth_box("hand", [0, 0, 100, 100]), "hand_left": truth_box("hand", [40, 0, 140, 100])}
    boxes = [model_box("hand", [30, 0, 130, 100]), model_box("hand", [0, 0, 100, 100]),
             model_box("hand", [2, 0, 102, 100])]
    box_of, on_no_target = box_of_each_target_that_counts(truth, boxes, HALF_OVERLAP)
    assert box_of == {"hand_right": 1, "hand_left": 0}
    assert on_no_target == [2], "a second box on a hand that already has one is left over"
