from dataclasses import dataclass

RIGHT, WRONG, ON_A_BOX_THAT_DOES_NOT_COUNT = "right", "wrong", "on a box that does not count"


@dataclass(frozen=True)
class Truth:
    class_name: str
    box: tuple[float, float, float, float]
    counts: bool = True
    visible_fraction: float | None = None


@dataclass(frozen=True)
class Detection:
    class_name: str
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class Judged:
    truth_was_found: tuple[bool, ...]
    detection_was: tuple[str, ...]


@dataclass
class Tally:
    true_boxes: int = 0
    found: int = 0
    detections: int = 0
    right: int = 0

    @property
    def found_share(self) -> float | None:
        return self.found / self.true_boxes if self.true_boxes else None

    @property
    def right_share(self) -> float | None:
        return self.right / self.detections if self.detections else None


def iou(a, b) -> float:
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    overlap = overlap_w * overlap_h
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def judged(truths: list[Truth], detections: list[Detection], min_iou: float) -> Judged:
    truth_was_found = [False] * len(truths)
    detection_was = [WRONG] * len(detections)
    most_confident_first = sorted(range(len(detections)), key=lambda d: -detections[d].confidence)
    for d in most_confident_first:
        detection = detections[d]
        overlapped = [(iou(truth.box, detection.box), t) for t, truth in enumerate(truths)
                      if truth.class_name == detection.class_name]
        overlapped = [(overlap, t) for overlap, t in overlapped if overlap >= min_iou]
        still_to_find = [(overlap, t) for overlap, t in overlapped if truths[t].counts and not truth_was_found[t]]
        if still_to_find:
            truth_was_found[max(still_to_find)[1]] = True
            detection_was[d] = RIGHT
        elif any(not truths[t].counts for _, t in overlapped):
            detection_was[d] = ON_A_BOX_THAT_DOES_NOT_COUNT
    return Judged(tuple(truth_was_found), tuple(detection_was))
