import math

import numpy as np


def pixels_with_their_centre_within(half_extent: float, of: float, pixels_in_the_image: int) -> slice:
    pixel_holding_the_centre = math.floor(of)
    first = min(math.ceil(of - half_extent - 0.5), pixel_holding_the_centre)
    last = max(math.floor(of + half_extent - 0.5), pixel_holding_the_centre)
    return slice(max(first, 0), max(min(last, pixels_in_the_image - 1) + 1, 0))


def position_or_none_without_depth(box, depth: np.ndarray, metres_per_depth_unit: float, K: np.ndarray,
                                   T_table_cam: np.ndarray, middle_fraction: float,
                                   min_valid_fraction: float) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    rows = pixels_with_their_centre_within(middle_fraction * (y1 - y0) / 2, yc, depth.shape[0])
    columns = pixels_with_their_centre_within(middle_fraction * (x1 - x0) / 2, xc, depth.shape[1])
    middle = depth[rows, columns]
    readings = middle[middle > 0]
    if readings.size == 0 or readings.size < min_valid_fraction * middle.size:
        return None
    z = float(np.median(readings)) * metres_per_depth_unit
    u, v = xc - 0.5, yc - 0.5
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
    return (np.asarray(T_table_cam) @ p_cam)[:3]
