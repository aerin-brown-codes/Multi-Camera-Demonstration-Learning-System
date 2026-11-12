from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxLimits:
    min_visible_fraction: float
    min_box_side_px: int


@dataclass(frozen=True)
class TargetBox:
    box: list[int] | None
    visible_fraction: float
    visible_px: int
    counts: bool


def x0_y0_x1_y1_with_x1_y1_one_past_the_last_pixel(pixels: np.ndarray) -> list[int] | None:
    columns, rows = np.flatnonzero(pixels.any(axis=0)), np.flatnonzero(pixels.any(axis=1))
    if columns.size == 0:
        return None
    return [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]


def target_box_or_none_when_out_of_view(visible_pixels: np.ndarray, pixels_when_alone: np.ndarray,
                                        limits: BoxLimits) -> TargetBox | None:
    alone_px = int(np.count_nonzero(pixels_when_alone))
    if alone_px == 0:
        return None
    visible_px = int(np.count_nonzero(visible_pixels))
    box = x0_y0_x1_y1_with_x1_y1_one_past_the_last_pixel(visible_pixels)
    fraction = visible_px / alone_px
    large_enough = box is not None and min(box[2] - box[0], box[3] - box[1]) >= limits.min_box_side_px
    return TargetBox(box, fraction, visible_px, large_enough and fraction >= limits.min_visible_fraction)
