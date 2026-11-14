import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import yaml
from feed.detections import DETECTIONS_TOPIC_PREFIX, ClassBoxConfidence, detections_of
from feed.frames import Frame, FramePairer
from feed.messages import (CAMERA_TOPIC_PREFIX, METRES_PER_DEPTH_UNIT, TRUTH_TOPIC, seconds_from_stamp,
                           stamp_from_fields)
from feed.pipe import Subscriber

from sim.run import DEFAULT_CONFIG

WINDOW = "camera feed"
WAITING_SCREEN = np.zeros((240, 480, 3), np.uint8)
ESC = 27
HEADER_HEIGHT_PX = 30
COUNTED_BOX_BGR = (0, 255, 0)
UNCOUNTED_BOX_BGR = (150, 150, 150)
MODEL_BOX_BGR = (255, 0, 255)
BOX_LABEL_HEIGHT_PX = 14
RATES_ARE_OVER_THE_LAST_S = 1.0


@dataclass
class KeptFrame:
    frame: Frame
    truth_boxes: dict
    model_boxes: list[ClassBoxConfidence] | None = None


@dataclass(frozen=True)
class Shown:
    frame: Frame
    truth_boxes: dict
    model_boxes: list[ClassBoxConfidence] | None
    frames_a_second: int
    frames_the_model_boxed_a_second: int | None


class WhatToShow:
    def __init__(self, frames_kept: int, detector_quiet_after_s: float):
        self._frames_kept = frames_kept
        self._detector_quiet_after_s = detector_quiet_after_s
        self._kept: dict[str, dict[tuple[int, int], KeptFrame]] = {}
        self._frame_times: dict[str, list[float]] = {}
        self._model_boxes_times: dict[str, list[float]] = {}
        self._last_model_boxes_at: dict[str, float] = {}

    def add_frame(self, frame: Frame, truth_boxes: dict, now: float) -> None:
        kept = self._kept.setdefault(frame.camera, {})
        kept.pop(frame.stamp, None)
        kept[frame.stamp] = KeptFrame(frame, truth_boxes)
        if len(kept) > self._frames_kept:
            del kept[next(iter(kept))]
        self._frame_times.setdefault(frame.camera, []).append(now)

    def add_model_boxes(self, camera: str, stamp: tuple[int, int], boxes: list[ClassBoxConfidence], now: float) -> None:
        the_frame_they_were_found_in = self._kept.get(camera, {}).get(stamp)
        if the_frame_they_were_found_in is None:
            return
        the_frame_they_were_found_in.model_boxes = boxes
        self._model_boxes_times.setdefault(camera, []).append(now)
        self._last_model_boxes_at[camera] = now

    def at(self, now: float) -> list[Shown]:
        shown = []
        for camera in sorted(self._kept):
            newest_last = list(self._kept[camera].values())
            with_model_boxes = [kept for kept in newest_last if kept.model_boxes is not None]
            last_model_boxes_at = self._last_model_boxes_at.get(camera)
            detector_is_quiet = (last_model_boxes_at is None
                                 or now - last_model_boxes_at > self._detector_quiet_after_s)
            kept = newest_last[-1] if detector_is_quiet or not with_model_boxes else with_model_boxes[-1]
            rates = []
            for times in (self._frame_times, self._model_boxes_times):
                times[camera] = [t for t in times.get(camera, []) if now - t < RATES_ARE_OVER_THE_LAST_S]
                rates.append(len(times[camera]))
            shown.append(Shown(kept.frame, kept.truth_boxes, kept.model_boxes, rates[0],
                               None if detector_is_quiet else rates[1]))
        return shown


def depth_to_color(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    metres = depth * METRES_PER_DEPTH_UNIT
    nearness = np.clip((far_m - metres) / (far_m - near_m), 0, 1)
    colored = cv2.applyColorMap((nearness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth == 0] = 0
    return colored


def outline_just_outside_its_pixels(img: np.ndarray, box, scale: float, color) -> tuple[int, int, int]:
    x0, y0, x1, y1 = box
    left, top, bottom = math.floor(x0 * scale) - 1, math.floor(y0 * scale) - 1, math.ceil(y1 * scale)
    cv2.rectangle(img, (left, top), (math.ceil(x1 * scale), bottom), color, 1)
    return left, top, bottom


def write_box_label(img: np.ndarray, text: str, left: int, baseline: int, color) -> None:
    cv2.putText(img, text, (max(left, 0), baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_boxes_just_outside_their_pixels(img: np.ndarray, boxes: dict, scale: float) -> None:
    for name, b in boxes.items():
        if b["box"] is None:
            continue
        color = COUNTED_BOX_BGR if b["counts"] else UNCOUNTED_BOX_BGR
        left, top, _ = outline_just_outside_its_pixels(img, b["box"], scale, color)
        label_above = max(top - 4, HEADER_HEIGHT_PX + BOX_LABEL_HEIGHT_PX)
        write_box_label(img, f"{name} {b['visible_fraction']:.2f}", left, label_above, color)


def draw_model_boxes_labelled_under(img: np.ndarray, boxes: list[ClassBoxConfidence], scale: float) -> None:
    for class_name, box, confidence in boxes:
        left, _, bottom = outline_just_outside_its_pixels(img, box, scale, MODEL_BOX_BGR)
        label_under = min(bottom + BOX_LABEL_HEIGHT_PX, img.shape[0] - 4)
        write_box_label(img, f"{class_name} {confidence:.2f}", left, label_under, MODEL_BOX_BGR)


def tile(shown: Shown, near_m: float, far_m: float, scale: float) -> np.ndarray:
    f = shown.frame
    img = np.vstack([cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), depth_to_color(f.depth, near_m, far_m)])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    colour_image = img[:img.shape[0] // 2]
    draw_boxes_just_outside_their_pixels(colour_image, shown.truth_boxes, scale)
    draw_model_boxes_labelled_under(colour_image, shown.model_boxes or [], scale)
    label = f"{f.camera}  t {seconds_from_stamp(f.stamp):.2f}s  {shown.frames_a_second} fps"
    if shown.frames_the_model_boxed_a_second is not None:
        label += f"  model {shown.frames_the_model_boxed_a_second} fps"
    cv2.rectangle(img, (0, 0), (img.shape[1], HEADER_HEIGHT_PX), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compose(tiles: list[np.ndarray]) -> np.ndarray:
    cols = math.ceil(math.sqrt(len(tiles)))
    h, w = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    padded += [np.zeros((h, w, 3), np.uint8)] * (-len(padded) % cols)
    return np.vstack([np.hstack(padded[i:i + cols]) for i in range(0, len(padded), cols)])


def main():
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    feed, view = cfg["feed"], cfg["view"]
    parser = argparse.ArgumentParser(description="Show every camera in the feed live, color above depth, with the "
                                                 "sim's truth boxes when it sends any and the model's boxes while "
                                                 "detect.live runs. Press q or Esc to quit.")
    parser.add_argument("--address", default=f"tcp://{feed['host']}:{feed['port']}")
    parser.add_argument("--detections-address", default=view["detections_address"])
    args = parser.parse_args()
    near_m, far_m = view["depth_display_m"]

    sub = Subscriber([args.address, args.detections_address],
                     [CAMERA_TOPIC_PREFIX, TRUTH_TOPIC, DETECTIONS_TOPIC_PREFIX], feed["queue_messages"])
    pairer = FramePairer()
    what_to_show = WhatToShow(view["frames_kept"], view["detector_quiet_after_s"])
    boxes_of_frames_not_yet_shown: dict[tuple[int, int], dict] = {}
    drawn = []
    print(f"view: frames from {args.address}, model boxes from {args.detections_address}  (q or Esc to quit)")
    cv2.namedWindow(WINDOW)
    cv2.imshow(WINDOW, WAITING_SCREEN)
    while True:
        msg = sub.recv(timeout_ms=50)
        while msg is not None:
            topic, fields, data = msg
            if topic == TRUTH_TOPIC:
                boxes_of_frames_not_yet_shown[stamp_from_fields(fields["stamp"])] = fields.get("boxes", {})
            elif topic.startswith(DETECTIONS_TOPIC_PREFIX):
                what_to_show.add_model_boxes(*detections_of(topic, fields), time.perf_counter())
            else:
                f = pairer.add(topic, fields, data)
                if f is not None:
                    boxes_of_frames_not_yet_shown = {s: b for s, b in boxes_of_frames_not_yet_shown.items()
                                                     if s >= f.stamp}
                    truth_boxes = boxes_of_frames_not_yet_shown.get(f.stamp, {}).get(f.camera, {})
                    what_to_show.add_frame(f, truth_boxes, time.perf_counter())
            msg = sub.recv(timeout_ms=0)
        shown = what_to_show.at(time.perf_counter())
        to_draw = [(s.frame.camera, s.frame.stamp, s.model_boxes is not None) for s in shown]
        if to_draw != drawn:
            drawn = to_draw
            cv2.imshow(WINDOW, compose([tile(s, near_m, far_m, view["scale"]) for s in shown]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ESC) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    sub.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
