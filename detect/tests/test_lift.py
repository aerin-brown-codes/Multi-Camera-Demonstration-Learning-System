import subprocess
import sys

import numpy as np
import pytest

from detect.lift import position_or_none_without_depth

WIDTH, HEIGHT = 64, 48
K = np.array([[50.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]])
MILLIMETRES = 0.001
CAMERA_COORDINATES = np.eye(4)
EXACT_M = 1e-9
WALL_MM, OTHER_MM = 1000, 400


def wall(depth_mm=WALL_MM):
    return np.full((HEIGHT, WIDTH), depth_mm, np.uint16)


def lifted(box, depth, T_table_cam=CAMERA_COORDINATES, middle_fraction=0.5, min_valid_fraction=0.5):
    return position_or_none_without_depth(box, depth, MILLIMETRES, K, T_table_cam, middle_fraction, min_valid_fraction)


def camera_posed_in_the_table_frame():
    forward = np.array([-0.5, 0.4, -0.6]) / np.linalg.norm([-0.5, 0.4, -0.6])
    right = np.cross(forward, [0.0, 0.0, 1.0]) / np.linalg.norm(np.cross(forward, [0.0, 0.0, 1.0]))
    T_table_cam = np.eye(4)
    T_table_cam[:3, :3] = np.column_stack([right, np.cross(forward, right), forward])
    T_table_cam[:3, 3] = [0.5, -0.4, 1.3]
    return T_table_cam


def test_a_wall_one_metre_away_under_a_box_centred_on_the_lens_centre_is_straight_ahead():
    centred_on_the_lens_centre = (22.0, 14.0, 42.0, 34.0)
    assert lifted(centred_on_the_lens_centre, wall()) == pytest.approx([0.0, 0.0, 1.0], abs=EXACT_M)


def test_a_box_covering_one_pixel_lifts_along_the_line_through_that_pixels_centre():
    column, row = 40, 10
    point = lifted((column, row, column + 1, row + 1), wall(2000))
    assert point == pytest.approx([(column - 31.5) * 2.0 / 50.0, (row - 23.5) * 2.0 / 60.0, 2.0], abs=EXACT_M)


def test_depth_is_the_distance_along_the_lens_axis_not_along_the_line_of_sight():
    far_from_the_lens_centre = (0.0, 0.0, 4.0, 4.0)
    assert lifted(far_from_the_lens_centre, wall())[2] == pytest.approx(1.0, abs=EXACT_M)


def test_a_point_put_on_the_table_and_seen_by_a_posed_camera_comes_back_where_it_was_put():
    T_table_cam = camera_posed_in_the_table_frame()
    column, row, depth_mm = 37, 29, 1234
    p_cam = [(column - 31.5) * 1.234 / 50.0, (row - 23.5) * 1.234 / 60.0, 1.234, 1.0]
    p_table = (T_table_cam @ p_cam)[:3]
    point = lifted((column, row, column + 1, row + 1), wall(depth_mm), T_table_cam)
    assert point == pytest.approx(p_table, abs=EXACT_M)
    the_pose_used_the_wrong_way_round = (np.linalg.inv(T_table_cam) @ p_cam)[:3]
    assert np.linalg.norm(point - the_pose_used_the_wrong_way_round) > 0.1


def test_the_middle_is_the_box_shrunk_about_its_centre_and_depths_outside_it_change_nothing():
    box = (10.0, 8.0, 50.0, 40.0)
    depth = np.full((HEIGHT, WIDTH), OTHER_MM, np.uint16)
    depth[16:32, 20:40] = WALL_MM
    assert lifted(box, depth)[2] == pytest.approx(1.0, abs=EXACT_M), "rows 16 to 31, columns 20 to 39 are the middle"
    for row, column, in_the_middle in ((16, 20, True), (31, 39, True), (15, 20, False), (16, 19, False),
                                       (32, 39, False), (31, 40, False)):
        the_only_reading = wall(0)
        the_only_reading[row, column] = WALL_MM
        point = lifted(box, the_only_reading, min_valid_fraction=0.0)
        assert (point is not None) == in_the_middle, f"row {row}, column {column}"


def test_other_depths_in_less_than_half_of_the_middle_change_nothing():
    box = (10.0, 8.0, 50.0, 40.0)
    depth = wall()
    depth[16:32, 20:29] = OTHER_MM
    assert lifted(box, depth)[2] == pytest.approx(1.0, abs=EXACT_M)
    depth[16:32, 20:31] = OTHER_MM
    assert lifted(box, depth)[2] == pytest.approx(OTHER_MM / 1000, abs=EXACT_M)


def test_with_an_even_count_the_depth_is_the_mean_of_the_two_middle_readings():
    depth = wall()
    depth[10, 40] = 1001
    assert lifted((40.0, 10.0, 42.0, 11.0), depth, middle_fraction=1.0)[2] == pytest.approx(1.0005, abs=EXACT_M)


def test_pixels_without_a_reading_are_dropped_and_too_few_readings_give_no_position():
    box = (10.0, 8.0, 50.0, 40.0)
    pixels_in_the_middle = 16 * 20
    for holes, gives_a_position in ((pixels_in_the_middle // 2, True), (pixels_in_the_middle // 2 + 1, False)):
        depth = wall()
        depth[16:32, 20:40].flat[:holes] = 0
        point = lifted(box, depth)
        assert (point is not None) == gives_a_position, f"{holes} of {pixels_in_the_middle} pixels without a reading"
        if gives_a_position:
            assert point[2] == pytest.approx(1.0, abs=EXACT_M), "a pixel without a reading must not pull the depth down"


def test_a_middle_with_no_reading_at_all_gives_no_position_whatever_the_fraction_asked_for():
    assert lifted((10.0, 8.0, 50.0, 40.0), wall(0), min_valid_fraction=0.0) is None


def test_a_box_at_the_images_edge_uses_only_the_pixels_inside_the_image():
    reaching_past_the_right_and_bottom_edge = (44.0, 36.0, 84.0, 60.0)
    depth = wall()
    depth[42:, 54:] = OTHER_MM
    point = lifted(reaching_past_the_right_and_bottom_edge, depth)
    assert point == pytest.approx([(63.5 - 31.5) * 0.4 / 50.0, (47.5 - 23.5) * 0.4 / 60.0, 0.4], abs=EXACT_M)


def test_a_box_whose_middle_lies_wholly_outside_the_image_gives_no_position():
    assert lifted((WIDTH + 10.0, 10.0, WIDTH + 30.0, 30.0), wall()) is None
    assert lifted((-30.0, 10.0, -10.0, 30.0), wall()) is None


def test_the_lift_loads_neither_feed_nor_zmq_nor_the_training_tool():
    code = "import sys, detect.lift\nprint(sorted({'feed', 'zmq', 'torch', 'ultralytics'} & set(sys.modules)))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
