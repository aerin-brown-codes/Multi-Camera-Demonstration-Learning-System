import numpy as np
import pytest
from feed.frames import Frame

from sim.lift_error import lift_error
from sim.view import (BOX_CENTRE_DOT_BGR, COUNTED_BOX_BGR, HEADER_HEIGHT_PX, MODEL_BOX_BGR, UNCOUNTED_BOX_BGR, Shown,
                      TruthOfTheLift, WhatToShow, position_and_error_under_each_model_box, tile,
                      where_a_point_shows_or_none_outside_the_picture)

WIDTH, HEIGHT = 300, 200
MIN_OVERLAP = 0.5
K = np.array([[250.0, 0, 149.5], [0, 240.0, 99.5], [0, 0, 1]])
BLACK_FRAME = Frame("cam0", (0, 0), np.zeros((HEIGHT, WIDTH, 3), np.uint8), np.zeros((HEIGHT, WIDTH), np.uint16),
                    np.eye(3))
FRAME_WITH_DEPTH = Frame("cam0", (0, 0), BLACK_FRAME.rgb, np.full((HEIGHT, WIDTH), 1000, np.uint16), np.eye(3))


def box(x0_y0_x1_y1, counts=True):
    return {"class": "cube", "box": x0_y0_x1_y1, "visible_fraction": 1.0, "visible_px": 1, "counts": counts}


def shown_with(boxes, model_boxes=None, positions=None, truth_of_the_lift=None, frame=BLACK_FRAME):
    return Shown(frame, boxes, model_boxes, frames_a_second=30, frames_the_model_boxed_a_second=None,
                 truth_of_the_lift=truth_of_the_lift, positions=positions)


def drawn(boxes, scale=1.0, **of_the_frame_shown):
    return tile(shown_with(boxes, **of_the_frame_shown), near_m=0.5, far_m=1.5, scale=scale, min_overlap=MIN_OVERLAP)


def colour_half(img):
    return img[:img.shape[0] // 2]


def depth_half(img):
    return img[img.shape[0] // 2:]


def drawn_on(depth_image):
    return (depth_image != depth_image[0, 0]).any(axis=-1)


def camera_looking_along_x_from(position):
    T_table_cam = np.eye(4)
    T_table_cam[:3, :3] = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    T_table_cam[:3, 3] = position
    return T_table_cam


def point_seen_at(column, row, depth_m, T_table_cam):
    in_camera_coordinates = [(column - K[0, 2]) * depth_m / K[0, 0], (row - K[1, 2]) * depth_m / K[1, 1], depth_m, 1.0]
    return (T_table_cam @ in_camera_coordinates)[:3].tolist()


def test_box_is_outlined_just_outside_its_pixels_on_the_colour_image():
    img = drawn({"cube": box([100, 80, 160, 130])})
    for row in (79, 130):
        assert (img[row, 99:161] == COUNTED_BOX_BGR).all(), f"row {row} should be the outline"
    for column in (99, 160):
        assert (img[79:131, column] == COUNTED_BOX_BGR).all(), f"column {column} should be the outline"
    assert not img[80:130, 100:160].any(), "the outline must not cover the target's own pixels"


def test_box_follows_the_tile_when_it_is_shrunk():
    img = drawn({"cube": box([100, 80, 160, 130])}, scale=0.5)
    assert (img[39, 49:81] == COUNTED_BOX_BGR).all() and (img[65, 49:81] == COUNTED_BOX_BGR).all()


def test_box_that_does_not_count_is_grey():
    img = drawn({"cube": box([100, 80, 160, 130], counts=False)})
    assert (img[79, 99:161] == UNCOUNTED_BOX_BGR).all()


def test_truth_boxes_stay_off_the_depth_image():
    reaching_the_bottom_edge = box([100, 150, 160, HEIGHT])
    fully_hidden = box(None, counts=False)
    img = drawn({"cube": reaching_the_bottom_edge, "hand_left": fully_hidden})
    assert (img[149, 99:161] == COUNTED_BOX_BGR).all()
    assert not img[HEIGHT:].any(), "truth boxes are drawn on the colour image only"


def test_model_box_is_outlined_in_its_own_colour_on_the_colour_image_and_on_the_depth_image():
    img = drawn({}, model_boxes=[("cube", (100.0, 80.0, 160.0, 130.0), 0.91)], frame=FRAME_WITH_DEPTH)
    for half, name in ((colour_half(img), "colour"), (depth_half(img), "depth")):
        assert (half[79, 99:161] == MODEL_BOX_BGR).all() and (half[130, 99:161] == MODEL_BOX_BGR).all(), name
        assert (half[79:131, 99] == MODEL_BOX_BGR).all() and (half[79:131, 160] == MODEL_BOX_BGR).all(), name
    assert img[131:150, 99:161].any(), "class and confidence are written under the box on the colour image"
    assert not img[HEADER_HEIGHT_PX + 1:79].any(), "nothing is written above a model box: that is where truth labels go"
    assert not drawn_on(depth_half(img))[131:].any(), "while no positions have come for the frame, nothing is written"


def test_label_of_a_model_box_at_the_bottom_edge_stays_on_the_colour_image():
    img = drawn({}, model_boxes=[("hand", (100.0, 150.0, 160.0, float(HEIGHT)), 0.5)])
    assert img[HEIGHT - 20:HEIGHT, 99:161].any()
    assert not depth_half(img)[:149].any(), "the depth image has the box's own outline and nothing of the label"


def test_where_the_model_agrees_with_truth_its_outline_covers_the_green_one_and_both_labels_show():
    img = drawn({"cube": box([100, 80, 160, 130])}, model_boxes=[("cube", (100.0, 80.0, 160.0, 130.0), 0.91)])
    assert (img[79, 99:161] == MODEL_BOX_BGR).all(), "green left showing would mean the model's box is off"
    assert img[60:79, 99:161].any() and img[131:150, 99:161].any()


MODEL_BOXES = [("cube", (100.0, 80.0, 160.0, 130.0), 0.91), ("hand", (200.0, 20.0, 280.0, 100.0), 0.8)]
CUBE_LIFTED_AT = (0, "cube", (0.07, 0.104, 0.789), 0.91)


def test_the_position_is_written_under_its_box_on_the_depth_image_and_no_depth_under_a_box_without_one():
    shown = shown_with({}, model_boxes=MODEL_BOXES, positions=[CUBE_LIFTED_AT], frame=FRAME_WITH_DEPTH)
    assert position_and_error_under_each_model_box(shown, MIN_OVERLAP) == [["0.07 0.10 0.79"], ["no depth"]]
    depth = depth_half(tile(shown, near_m=0.5, far_m=1.5, scale=1.0, min_overlap=MIN_OVERLAP))
    written = drawn_on(depth)
    assert written[131:150, 99:220].any() and written[101:120, 199:270].any(), "one line under each box"
    assert not written[150:].any(), "a box with a position and no truth has one line, not two"
    assert (depth[131:150][written[131:150]] <= depth[0, 0]).all(), \
        "the text only darkens the depth colour: black, with no edge in another colour"


def test_boxes_whose_positions_have_not_come_have_no_text_under_them():
    shown = shown_with({}, model_boxes=MODEL_BOXES, positions=None)
    assert position_and_error_under_each_model_box(shown, MIN_OVERLAP) == [[], []]


def test_ids_are_places_in_the_boxes_list_so_a_missing_first_id_leaves_the_second_box_its_own_position():
    shown = shown_with({}, model_boxes=MODEL_BOXES, positions=[(1, "hand", (0.2, -0.1, 0.8), 0.8)])
    assert position_and_error_under_each_model_box(shown, MIN_OVERLAP) == [["no depth"], ["0.20 -0.10 0.80"]]


def test_two_lines_under_a_box_at_the_bottom_edge_both_stay_on_the_depth_image():
    T_table_cam = camera_looking_along_x_from([0.0, 0.0, 1.0])
    truth = TruthOfTheLift({"cube": point_seen_at(130, 175, 1.0, T_table_cam)}, T_table_cam)
    at_the_bottom_edge = [100, 150, 160, HEIGHT]
    img = drawn({"cube": box(at_the_bottom_edge)}, model_boxes=[("cube", tuple(map(float, at_the_bottom_edge)), 0.9)],
                positions=[(0, "cube", (0.97, 0.0, 1.0), 0.9)], truth_of_the_lift=truth, frame=FRAME_WITH_DEPTH)
    assert img.shape[0] == 2 * HEIGHT
    under_the_colour_label = drawn_on(depth_half(img))[HEIGHT - 2 * 14 - 4:, 99:220]
    assert under_the_colour_label[:14].any() and under_the_colour_label[14:].any()


def test_a_truth_point_is_drawn_where_the_camera_sees_it():
    T_table_cam = camera_looking_along_x_from([-0.4, 0.3, 1.1])
    for column, row in ((0, 0), (130, 105), (WIDTH - 1, HEIGHT - 1)):
        point = point_seen_at(column, row, 0.83, T_table_cam)
        assert where_a_point_shows_or_none_outside_the_picture(point, T_table_cam, K, 1.0, (HEIGHT, WIDTH)) == \
            (column, row)
        assert where_a_point_shows_or_none_outside_the_picture(point, T_table_cam, K, 0.5, (HEIGHT // 2, WIDTH // 2)) \
            == (column // 2, row // 2), "on a shrunk tile it is the shrunk pixel"
    for outside in (point_seen_at(-1, 50, 0.83, T_table_cam), point_seen_at(50, HEIGHT, 0.83, T_table_cam),
                    point_seen_at(130, 105, -0.83, T_table_cam)):
        assert where_a_point_shows_or_none_outside_the_picture(outside, T_table_cam, K, 1.0, (HEIGHT, WIDTH)) is None


def test_with_truth_a_green_dot_marks_each_counting_targets_point_and_a_black_dot_each_box_centre():
    T_table_cam = camera_looking_along_x_from([-0.4, 0.3, 1.1])
    truth = TruthOfTheLift({"cube": point_seen_at(129, 106, 0.83, T_table_cam),
                            "hand_left": point_seen_at(40, 40, 0.83, T_table_cam)}, T_table_cam)
    frame_with_a_lens = Frame("cam0", (0, 0), BLACK_FRAME.rgb, BLACK_FRAME.depth, K)
    img = drawn({"cube": box([100, 80, 160, 130]), "hand_left": box([20, 20, 60, 60], counts=False)},
                model_boxes=MODEL_BOXES[:1], positions=[CUBE_LIFTED_AT], truth_of_the_lift=truth,
                frame=frame_with_a_lens)
    depth = depth_half(img)
    assert (depth[106, 127:129] == COUNTED_BOX_BGR).all() and (depth[107:109, 129] == COUNTED_BOX_BGR).all(), \
        "a dot reaching 2 pixels from the pixel the truth point shows at"
    assert (depth[104:107, 130] == BOX_CENTRE_DOT_BGR).all() and (depth[105, 129:132] == BOX_CENTRE_DOT_BGR).all(), \
        "a dot reaching 1 pixel from the box's centre (130, 105), on top of the truth point's dot"
    assert not (depth[30:51, 30:51] == COUNTED_BOX_BGR).all(axis=-1).any(), "no dot for a target that does not count"
    assert not (colour_half(img)[100:117, 119:136] == COUNTED_BOX_BGR).all(axis=-1).any(), \
        "the dots are on the depth image only"


def test_the_error_line_under_a_paired_box_is_the_reports_own_numbers():
    T_table_cam = camera_looking_along_x_from([-0.4, 0.3, 1.1])
    truth_point = [0.4, 0.3, 1.1]
    lifted_point = (0.4 - 0.027, 0.3 + 0.002, 1.1)
    error = lift_error(T_table_cam[:3, 3], truth_point, lifted_point)
    shown = shown_with({"cube": box([100, 80, 160, 130])}, model_boxes=MODEL_BOXES,
                       positions=[(0, "cube", lifted_point, 0.91), (1, "hand", (0.2, -0.1, 0.8), 0.8)],
                       truth_of_the_lift=TruthOfTheLift({"cube": truth_point}, T_table_cam))
    assert position_and_error_under_each_model_box(shown, MIN_OVERLAP) == [
        ["0.37 0.30 1.10", f"side {error.sideways_mm:.0f} along {error.along_mm:.0f}"], ["0.20 -0.10 0.80"]], \
        "the hand's box pairs with no target, so it has a position and no error"
    assert (error.sideways_mm, error.along_mm) == pytest.approx((2.0, -27.0), abs=1e-9), "so it reads side 2 along -27"


def test_a_paired_box_without_a_position_has_no_error_line():
    T_table_cam = camera_looking_along_x_from([-0.4, 0.3, 1.1])
    shown = shown_with({"cube": box([100, 80, 160, 130])}, model_boxes=MODEL_BOXES[:1], positions=[],
                       truth_of_the_lift=TruthOfTheLift({"cube": [0.4, 0.3, 1.1]}, T_table_cam))
    assert position_and_error_under_each_model_box(shown, MIN_OVERLAP) == [["no depth"]]


def test_without_truth_there_are_no_dots_and_no_error_line():
    shown = shown_with({"cube": box([100, 80, 160, 130])}, model_boxes=MODEL_BOXES[:1], positions=[CUBE_LIFTED_AT],
                       frame=FRAME_WITH_DEPTH)
    assert position_and_error_under_each_model_box(shown, MIN_OVERLAP) == [["0.07 0.10 0.79"]]
    depth = depth_half(tile(shown, near_m=0.5, far_m=1.5, scale=1.0, min_overlap=MIN_OVERLAP))
    assert not drawn_on(depth)[80:130, 100:160].any(), "nothing is drawn inside the box"
    assert not (depth == COUNTED_BOX_BGR).all(axis=-1).any()


def frame(camera, number):
    return Frame(camera, (0, number), np.full((HEIGHT, WIDTH, 3), number, np.uint8),
                 np.zeros((HEIGHT, WIDTH), np.uint16), np.eye(3))


def what_to_show_with_frames(numbers, cameras=("cam0",), frames_kept=8):
    what = WhatToShow(frames_kept, detector_quiet_after_s=2.0)
    for number in numbers:
        for camera in cameras:
            what.add_frame(frame(camera, number), {"cube": box([number, 0, 50, 50])}, now=number / 10)
    return what


def numbers_shown(what, now):
    return {shown.frame.camera: shown.frame.stamp[1] for shown in what.at(now)}


def test_with_no_detector_every_tile_shows_its_newest_frame_and_no_model_rate():
    what = what_to_show_with_frames(range(1, 6), cameras=("cam1", "cam0"))
    shown = what.at(now=0.5)
    assert [s.frame.camera for s in shown] == ["cam0", "cam1"], "tiles are in camera name order"
    for s in shown:
        assert (s.frame.stamp, s.model_boxes, s.frames_the_model_boxed_a_second) == ((0, 5), None, None)
    assert [s.frames_a_second for s in shown] == [5, 5]


def test_boxes_for_an_older_kept_frame_bring_that_frame_back_with_its_own_exact_boxes():
    what = what_to_show_with_frames(range(1, 6))
    model_boxes = [("cube", (3.0, 0.0, 50.0, 50.0), 0.9)]
    what.add_model_boxes("cam0", (0, 3), model_boxes, now=0.5)
    (shown,) = what.at(now=0.5)
    assert shown.frame.stamp == (0, 3) and shown.frame.rgb[0, 0, 0] == 3
    assert shown.truth_boxes["cube"]["box"] == [3, 0, 50, 50], "the exact boxes must be the shown frame's own"
    assert shown.model_boxes == model_boxes and shown.frames_the_model_boxed_a_second == 1


def test_a_frame_the_model_found_nothing_in_is_still_shown_as_boxed():
    what = what_to_show_with_frames(range(1, 6))
    what.add_model_boxes("cam0", (0, 4), [], now=0.5)
    (shown,) = what.at(now=0.5)
    assert shown.frame.stamp == (0, 4) and shown.model_boxes == []


def test_the_newest_frame_that_has_boxes_is_shown_whatever_order_the_boxes_came_in():
    what = what_to_show_with_frames(range(1, 6))
    for number in (4, 2):
        what.add_model_boxes("cam0", (0, number), [], now=0.5)
    assert numbers_shown(what, now=0.5) == {"cam0": 4}


def test_boxes_for_a_frame_that_is_no_longer_kept_change_nothing():
    what = what_to_show_with_frames(range(1, 12), frames_kept=8)
    what.add_model_boxes("cam0", (0, 3), [("cube", (0.0, 0.0, 50.0, 50.0), 0.9)], now=1.1)
    (shown,) = what.at(now=1.1)
    assert (shown.frame.stamp, shown.model_boxes, shown.frames_the_model_boxed_a_second) == ((0, 11), None, None)


def test_after_the_quiet_time_the_newest_frame_shows_again():
    what = what_to_show_with_frames(range(1, 6))
    what.add_model_boxes("cam0", (0, 3), [], now=0.5)
    assert numbers_shown(what, now=2.5) == {"cam0": 3}
    (shown,) = what.at(now=2.6)
    assert shown.frame.stamp == (0, 5) and shown.frames_the_model_boxed_a_second is None


def test_a_camera_that_never_had_boxes_shows_its_newest_frame_while_another_shows_its_boxed_one():
    what = what_to_show_with_frames(range(1, 6), cameras=("cam0", "cam1"))
    what.add_model_boxes("cam1", (0, 3), [], now=0.5)
    assert numbers_shown(what, now=0.5) == {"cam0": 5, "cam1": 3}


def test_a_stamp_that_comes_again_after_the_sim_restarted_is_a_new_frame_without_the_old_ones_boxes():
    what = what_to_show_with_frames(range(1, 4))
    what.add_model_boxes("cam0", (0, 1), [("cube", (0.0, 0.0, 50.0, 50.0), 0.9)], now=0.3)
    what.add_frame(frame("cam0", 1), {}, now=0.4)
    (shown,) = what.at(now=0.4)
    assert (shown.frame.stamp, shown.model_boxes) == ((0, 1), None)


def test_positions_are_shown_on_the_frame_they_were_lifted_in_and_on_no_other():
    what = what_to_show_with_frames(range(1, 6))
    what.add_model_boxes("cam0", (0, 3), [("cube", (3.0, 0.0, 50.0, 50.0), 0.9)], now=0.5)
    (shown,) = what.at(now=0.5)
    assert shown.frame.stamp == (0, 3) and shown.positions is None, "boxes have come, positions not yet"
    what.add_positions("cam0", (0, 3), [CUBE_LIFTED_AT])
    what.add_positions("cam0", (0, 99), [CUBE_LIFTED_AT])
    (shown,) = what.at(now=0.5)
    assert shown.frame.stamp == (0, 3) and shown.positions == [CUBE_LIFTED_AT]
    what.add_model_boxes("cam0", (0, 4), [], now=0.6)
    (shown,) = what.at(now=0.6)
    assert shown.frame.stamp == (0, 4) and shown.positions is None, "another frame never borrows these positions"


def test_the_truth_of_the_lift_stays_with_its_own_frame():
    what = WhatToShow(frames_kept=8, detector_quiet_after_s=2.0)
    truth = TruthOfTheLift({"cube": [0.0, 0.0, 0.8]}, np.eye(4))
    what.add_frame(frame("cam0", 1), {}, now=0.1, truth_of_the_lift=truth)
    what.add_frame(frame("cam0", 2), {}, now=0.2)
    what.add_model_boxes("cam0", (0, 1), [], now=0.2)
    (shown,) = what.at(now=0.2)
    assert shown.frame.stamp == (0, 1) and shown.truth_of_the_lift is truth
    (shown,) = what.at(now=5.0)
    assert shown.frame.stamp == (0, 2) and shown.truth_of_the_lift is None
