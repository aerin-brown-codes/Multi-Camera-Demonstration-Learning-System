import numpy as np

from sim.feed import Frame
from sim.view import COUNTED_BOX_BGR, UNCOUNTED_BOX_BGR, tile

WIDTH, HEIGHT = 300, 200
BLACK_FRAME = Frame("cam0", 0, 0.0, np.zeros((HEIGHT, WIDTH, 3), np.uint8), np.zeros((HEIGHT, WIDTH), np.uint16),
                    np.eye(3), 0.001)


def box(x0_y0_x1_y1, counts=True):
    return {"class": "cube", "box": x0_y0_x1_y1, "visible_fraction": 1.0, "visible_px": 1, "counts": counts}


def drawn(boxes, scale=1.0):
    return tile(BLACK_FRAME, boxes, fps=30, near_m=0.5, far_m=1.5, scale=scale)


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
