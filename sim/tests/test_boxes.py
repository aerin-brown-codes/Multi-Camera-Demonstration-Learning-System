import numpy as np

from sim.boxes import BoxLimits, target_box_or_none_when_out_of_view

LIMITS = BoxLimits(min_visible_fraction=0.10, min_box_side_px=8)
IMAGE = (60, 120)


def pixels(rows: slice, columns: slice) -> np.ndarray:
    image = np.zeros(IMAGE, bool)
    image[rows, columns] = True
    return image


def test_box_starts_at_the_first_pixel_and_ends_one_past_the_last():
    target = pixels(slice(20, 35), slice(40, 70))
    b = target_box_or_none_when_out_of_view(target, target, LIMITS)
    assert b.box == [40, 20, 70, 35], "box is [x0, y0, x1, y1] with x1 - x0 the width in pixels"
    assert (b.visible_px, b.visible_fraction, b.counts) == (15 * 30, 1.0, True)


def test_half_hidden_target_is_half_visible_and_its_box_shrinks_to_what_shows():
    alone = pixels(slice(20, 40), slice(40, 80))
    b = target_box_or_none_when_out_of_view(alone & pixels(slice(None), slice(60, None)), alone, LIMITS)
    assert b.visible_fraction == 0.5 and b.box == [60, 20, 80, 40] and b.counts


def test_target_cut_by_the_image_edge_but_not_hidden_is_fully_visible():
    at_the_corner = pixels(slice(0, 30), slice(100, None))
    b = target_box_or_none_when_out_of_view(at_the_corner, at_the_corner, LIMITS)
    assert b.visible_fraction == 1.0 and b.box == [100, 0, 120, 30]


def test_box_counts_at_the_visible_fraction_limit_and_not_just_under_it():
    alone = pixels(slice(10, 20), slice(10, 110))
    a_tenth = pixels(slice(10, 20), slice(10, 20))
    at_limit = target_box_or_none_when_out_of_view(a_tenth, alone, LIMITS)
    assert at_limit.visible_fraction == 0.10 and at_limit.counts

    one_pixel_fewer = a_tenth.copy()
    one_pixel_fewer[15, 15] = False
    just_under = target_box_or_none_when_out_of_view(one_pixel_fewer, alone, LIMITS)
    assert just_under.box == at_limit.box and just_under.visible_px == 99
    assert not just_under.counts, "99 of 1000 pixels visible is under the 10% limit"


def test_box_counts_at_the_side_limit_and_not_just_under_it():
    for rows, columns, counts in [(slice(10, 18), slice(10, 50), True), (slice(10, 17), slice(10, 50), False),
                                  (slice(10, 50), slice(10, 18), True), (slice(10, 50), slice(10, 17), False)]:
        target = pixels(rows, columns)
        b = target_box_or_none_when_out_of_view(target, target, LIMITS)
        assert b.counts == counts, f"box {b.box} against a shorter-side limit of {LIMITS.min_box_side_px} px"
        assert b.visible_fraction == 1.0, "a box that does not count is still reported in full"


def test_fully_hidden_target_is_reported_without_a_box():
    b = target_box_or_none_when_out_of_view(np.zeros(IMAGE, bool), pixels(slice(20, 40), slice(40, 80)), LIMITS)
    assert (b.box, b.visible_fraction, b.visible_px, b.counts) == (None, 0.0, 0, False)


def test_target_that_would_cover_no_pixel_even_alone_is_out_of_view_and_not_reported():
    nothing = np.zeros(IMAGE, bool)
    assert target_box_or_none_when_out_of_view(nothing, nothing, LIMITS) is None
