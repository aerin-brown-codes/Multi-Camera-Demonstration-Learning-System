import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from feed.detections import DETECTIONS_TOPIC_PREFIX, END_OF_A_POSITIONS_TOPIC, detections_of, positions_of
from feed.messages import TRUTH_TOPIC, stamp_from_fields
from feed.pipe import Subscriber

from sim.lift_error import LiftError, box_of_each_target_that_counts, lift_error
from sim.run import DEFAULT_CONFIG

MISSED, NO_DEPTH, LIFTED = "missed", "no depth", "lifted"
WHAT_THE_TRUTH_MUST_CARRY = ("pose", "boxes", "target_points")
COLUMNS = ("target  visible      views  missed  no depth   sideways mm            along mm\n"
           "                                               median   90%  worst    median  10% to 90%")
NO_ERRORS = f"{'-':>6} {'-':>5} {'-':>6}    {'-':>6}  -"


@dataclass(frozen=True)
class Sample:
    pose: str
    camera: str
    target: str
    class_name: str
    visible_fraction: float
    outcome: str
    error: LiftError | None = None


@dataclass(frozen=True)
class View:
    samples: list[Sample]
    boxes_on_no_target: int


def view_of(truth: dict, camera: str, model_boxes: list, positions: list, min_overlap: float) -> View:
    truth_boxes = truth["boxes"][camera]
    box_of, on_no_target = box_of_each_target_that_counts(truth_boxes, model_boxes, min_overlap)
    point_of_box = {int(box_id): point for box_id, _, point, _ in positions}
    camera_position = np.array(truth["cameras"][camera])[:3, 3]
    samples = []
    for target, box in box_of.items():
        outcome = MISSED if box is None else LIFTED if box in point_of_box else NO_DEPTH
        error = lift_error(camera_position, truth["target_points"][target], point_of_box[box]) \
            if outcome == LIFTED else None
        samples.append(Sample(truth["pose"]["name"], camera, target, truth_boxes[target]["class"],
                              truth_boxes[target]["visible_fraction"], outcome, error))
    return View(samples, len(on_no_target))


class Record:
    def __init__(self, min_overlap: float, frames_kept: int):
        self._min_overlap = min_overlap
        self._frames_kept = frames_kept
        self._truth_and_boxes_of: dict[tuple[int, int], tuple[dict, dict[str, list]]] = {}
        self.views: dict[tuple[int, str], View] = {}
        self.cameras: list[str] = []
        self.pose_count = 0
        self.cameras_that_sent_positions: set[str] = set()

    def add_truth(self, truth: dict) -> None:
        stamp = stamp_from_fields(truth["stamp"])
        self._truth_and_boxes_of.pop(stamp, None)
        self._truth_and_boxes_of[stamp] = truth, {}
        if len(self._truth_and_boxes_of) > self._frames_kept:
            del self._truth_and_boxes_of[next(iter(self._truth_and_boxes_of))]
        self.cameras, self.pose_count = sorted(truth["cameras"]), truth["pose"]["count"]

    def add_boxes(self, camera: str, stamp: tuple[int, int], boxes: list) -> None:
        if stamp in self._truth_and_boxes_of:
            self._truth_and_boxes_of[stamp][1][camera] = boxes

    def add_positions(self, camera: str, stamp: tuple[int, int], positions: list) -> None:
        self.cameras_that_sent_positions.add(camera)
        truth, boxes_of = self._truth_and_boxes_of.get(stamp, (None, {}))
        if camera in boxes_of and camera in truth["cameras"]:
            self.views[truth["pose"]["index"], camera] = view_of(truth, camera, boxes_of[camera], positions,
                                                                 self._min_overlap)

    def poses_not_seen_on_every_camera(self) -> list[int]:
        return [pose for pose in range(self.pose_count)
                if any((pose, camera) not in self.views for camera in self.cameras)]

    def has_every_pose_on_every_camera(self) -> bool:
        return self.pose_count > 0 and not self.poses_not_seen_on_every_camera()

    def samples(self) -> list[Sample]:
        return [sample for _, view in sorted(self.views.items()) for sample in view.samples]


def require_poses_and_targets(truth: dict) -> None:
    missing = [what for what in WHAT_THE_TRUTH_MUST_CARRY if what not in truth]
    if missing:
        raise SystemExit(f"check_lift: the sim's truth carries no {' and no '.join(missing)}, so there is nothing to "
                         f"measure. Its scene must list poses and targets: sim.run --scene cube_hands")


def bands_of(edges: list[float]) -> list[tuple[str, float, float, bool]]:
    bands = [(f"{100 * low:.0f} to {100 * high:.0f}%", low, high, high == edges[-1])
             for low, high in zip(edges, edges[1:])]
    return bands[::-1]


def in_band(sample: Sample, low: float, high: float, holds_its_upper_edge: bool) -> bool:
    return low <= sample.visible_fraction < high or (holds_its_upper_edge and sample.visible_fraction == high)


def row(first: str, second: str, samples: list[Sample]) -> str:
    errors = [sample.error for sample in samples if sample.outcome == LIFTED]
    counts = (f"{first:<7} {second:<11} {len(samples):>6}  {sum(s.outcome == MISSED for s in samples):>6}  "
              f"{sum(s.outcome == NO_DEPTH for s in samples):>8}   ")
    if not errors:
        return counts + NO_ERRORS
    sideways, along = [e.sideways_mm for e in errors], [e.along_mm for e in errors]
    nearest, farthest = np.percentile(along, [10, 90])
    return counts + (f"{np.median(sideways):6.1f} {np.percentile(sideways, 90):5.1f} {max(sideways):6.1f}    "
                     f"{np.median(along):6.0f}  {nearest:.0f} to {farthest:.0f}")


def report(record: Record, visible_fraction: list[float], worst_listed: int) -> str:
    samples, bands = record.samples(), bands_of(visible_fraction)
    classes = sorted({sample.class_name for sample in samples})
    lines = [f"lift on the sim feed: {record.pose_count} poses, {len(record.cameras)} cameras", "", COLUMNS]
    for class_name in classes:
        of_the_class = [sample for sample in samples if sample.class_name == class_name]
        lines.append(row(class_name, "all", of_the_class))
        lines += [row(class_name, label, [s for s in of_the_class if in_band(s, *edges)]) for label, *edges in bands]

    most_visible, *edges = bands[0]
    lines += ["", f"per camera, targets {most_visible} visible:"]
    lines += [row(class_name, camera, [s for s in samples if s.class_name == class_name and s.camera == camera
                                       and in_band(s, *edges)])
              for camera in record.cameras for class_name in classes]

    lifted = sorted((s for s in samples if s.outcome == LIFTED), key=lambda s: -s.error.distance_mm)
    lines += ["", f"worst {worst_listed} by distance:"]
    lines += [f"  {s.pose:<28} {s.camera:<5} {s.target:<11} {100 * s.visible_fraction:3.0f}% visible   "
              f"sideways {s.error.sideways_mm:5.1f}   along {s.error.along_mm:5.0f}   "
              f"distance {s.error.distance_mm:5.1f} mm" for s in lifted[:worst_listed]]
    lines += ["", "boxes on no target that counts: "
              f"{sum(view.boxes_on_no_target for view in record.views.values())}"]

    not_seen = record.poses_not_seen_on_every_camera()
    if not_seen:
        lines += ["", f"stopped early: {len(not_seen)} of {record.pose_count} poses were not seen on every camera "
                      f"(poses number {', '.join(str(pose) for pose in not_seen)}, counted from 0)"]
    sent_no_positions = [camera for camera in record.cameras if camera not in record.cameras_that_sent_positions]
    if sent_no_positions:
        lines.append(f"no positions came for {', '.join(sent_no_positions)}: detect.live has no camera pose for them")
    return "\n".join(lines)


def progress_line(record: Record, address: str, detections_address: str) -> str:
    if not record.pose_count:
        return f"check_lift: waiting for truth from {address} (sim.run should be running)"
    if not record.cameras_that_sent_positions:
        return f"check_lift: waiting for positions from {detections_address} (detect.live should be running)"
    seen = record.pose_count - len(record.poses_not_seen_on_every_camera())
    return f"check_lift: {seen} of {record.pose_count} poses seen on every camera"


def main():
    parser = argparse.ArgumentParser(
        description="Measure how far the positions detect.live publishes are from where the sim says the cube and "
                    "hands really are. Run it beside sim.run and detect.live. It prints one table when every pose "
                    "has been seen on every camera, or what it has so far on Ctrl+C. Nothing is saved.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--address")
    parser.add_argument("--detections-address")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    feed, check = cfg["feed"], cfg["check_lift"]
    address = args.address or f"tcp://{feed['host']}:{feed['port']}"
    detections_address = args.detections_address or cfg["view"]["detections_address"]

    sub = Subscriber([address, detections_address], [TRUTH_TOPIC, DETECTIONS_TOPIC_PREFIX], feed["queue_messages"])
    record = Record(check["min_overlap"], check["frames_kept"])
    print(f"check_lift: truth from {address}, boxes and positions from {detections_address}  "
          f"(Ctrl+C to print what it has so far)")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    last_progress = time.perf_counter()
    try:
        while not record.has_every_pose_on_every_camera():
            message = sub.recv(timeout_ms=200)
            if message is not None:
                topic, fields, _ = message
                if topic == TRUTH_TOPIC:
                    require_poses_and_targets(fields)
                    record.add_truth(fields)
                elif topic.endswith(END_OF_A_POSITIONS_TOPIC):
                    record.add_positions(*positions_of(topic, fields))
                else:
                    record.add_boxes(*detections_of(topic, fields))
            if time.perf_counter() - last_progress >= check["progress_every_s"]:
                print(progress_line(record, address, detections_address))
                last_progress = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
    print()
    print(report(record, check["visible_fraction"], check["worst_listed"]))


if __name__ == "__main__":
    main()
