from dataclasses import dataclass

import numpy as np

MM_PER_M = 1000


@dataclass(frozen=True)
class LiftError:
    sideways_mm: float
    along_mm: float
    distance_mm: float


def overlap_as_intersection_over_union(a, b) -> float:
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    overlap = overlap_w * overlap_h
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def box_of_each_target_that_counts(truth_boxes: dict, model_boxes: list, min_overlap: float
                                   ) -> tuple[dict[str, int | None], list[int]]:
    counting = {name: b for name, b in truth_boxes.items() if b["counts"]}
    largest_overlap_first = sorted(
        ((overlap_as_intersection_over_union(b["box"], box), name, place)
         for name, b in counting.items() for place, (class_name, box, _) in enumerate(model_boxes)
         if class_name == b["class"]), reverse=True)
    box_of: dict[str, int | None] = dict.fromkeys(counting)
    paired = set()
    for overlap, name, place in largest_overlap_first:
        if overlap >= min_overlap and box_of[name] is None and place not in paired:
            box_of[name] = place
            paired.add(place)
    return box_of, [place for place in range(len(model_boxes)) if place not in paired]


def lift_error(camera_position, truth_point, lifted_point) -> LiftError:
    camera_position, truth_point = np.asarray(camera_position, float), np.asarray(truth_point, float)
    off = np.asarray(lifted_point, float) - truth_point
    away_from_the_camera = (truth_point - camera_position) / np.linalg.norm(truth_point - camera_position)
    along = off @ away_from_the_camera
    sideways = np.linalg.norm(off - along * away_from_the_camera)
    return LiftError(MM_PER_M * float(sideways), MM_PER_M * float(along), MM_PER_M * float(np.linalg.norm(off)))
