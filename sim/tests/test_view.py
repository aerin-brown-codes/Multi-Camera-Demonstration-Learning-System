import numpy as np
from feed.frames import Frame

from sim.view import (COUNTED_BOX_BGR, HEADER_HEIGHT_PX, MODEL_BOX_BGR, UNCOUNTED_BOX_BGR, Shown, WhatToShow,
                      tile)

WIDTH, HEIGHT = 300, 200
BLACK_FRAME = Frame("cam0", (0, 0), np.zeros((HEIGHT, WIDTH, 3), np.uint8), np.zeros((HEIGHT, WIDTH), np.uint16),
                    np.eye(3))


def box(x0_y0_x1_y1, counts=True):
    return {"class": "cube", "box": x0_y0_x1_y1, "visible_fraction": 1.0, "visible_px": 1, "counts": counts}


def drawn(boxes, scale=1.0, model_boxes=None):
    shown = Shown(BLACK_FRAME, boxes, model_boxes, frames_a_second=30, frames_the_model_boxed_a_second=None)
    return tile(shown, near_m=0.5, far_m=1.5, scale=scale)


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


def test_boxes_stay_off_the_depth_image():
    reaching_the_bottom_edge = box([100, 150, 160, HEIGHT])
    fully_hidden = box(None, counts=False)
    img = drawn({"cube": reaching_the_bottom_edge, "hand_left": fully_hidden})
    assert (img[149, 99:161] == COUNTED_BOX_BGR).all()
    assert not img[HEIGHT:].any(), "boxes are drawn on the colour image only"


def test_model_box_is_outlined_in_its_own_colour_on_the_colour_image_only_with_its_label_under_it():
    img = drawn({}, model_boxes=[("cube", (100.0, 80.0, 160.0, 130.0), 0.91)])
    assert (img[79, 99:161] == MODEL_BOX_BGR).all() and (img[130, 99:161] == MODEL_BOX_BGR).all()
    assert (img[79:131, 99] == MODEL_BOX_BGR).all() and (img[79:131, 160] == MODEL_BOX_BGR).all()
    assert img[131:150, 99:161].any(), "class and confidence are written under the box"
    assert not img[HEADER_HEIGHT_PX + 1:79].any(), "nothing is written above a model box: that is where truth labels go"
    assert not img[HEIGHT:].any(), "model boxes are drawn on the colour image only"


def test_label_of_a_model_box_at_the_bottom_edge_stays_on_the_colour_image():
    img = drawn({}, model_boxes=[("hand", (100.0, 150.0, 160.0, float(HEIGHT)), 0.5)])
    assert img[HEIGHT - 20:HEIGHT, 99:161].any() and not img[HEIGHT:].any()


def test_where_the_model_agrees_with_truth_its_outline_covers_the_green_one_and_both_labels_show():
    img = drawn({"cube": box([100, 80, 160, 130])}, model_boxes=[("cube", (100.0, 80.0, 160.0, 130.0), 0.91)])
    assert (img[79, 99:161] == MODEL_BOX_BGR).all(), "green left showing would mean the model's box is off"
    assert img[60:79, 99:161].any() and img[131:150, 99:161].any()


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
