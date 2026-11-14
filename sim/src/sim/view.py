import argparse
import math
import time

import cv2
import numpy as np
import yaml
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
BOX_LABEL_HEIGHT_PX = 14


def depth_to_color(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    metres = depth * METRES_PER_DEPTH_UNIT
    nearness = np.clip((far_m - metres) / (far_m - near_m), 0, 1)
    colored = cv2.applyColorMap((nearness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth == 0] = 0
    return colored


def draw_boxes_just_outside_their_pixels(img: np.ndarray, boxes: dict, scale: float) -> None:
    for name, b in boxes.items():
        if b["box"] is None:
            continue
        x0, y0, x1, y1 = b["box"]
        left, top = math.floor(x0 * scale) - 1, math.floor(y0 * scale) - 1
        color = COUNTED_BOX_BGR if b["counts"] else UNCOUNTED_BOX_BGR
        cv2.rectangle(img, (left, top), (math.ceil(x1 * scale), math.ceil(y1 * scale)), color, 1)
        label_baseline = max(top - 4, HEADER_HEIGHT_PX + BOX_LABEL_HEIGHT_PX)
        cv2.putText(img, f"{name} {b['visible_fraction']:.2f}", (max(left, 0), label_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def tile(f: Frame, boxes: dict, fps: float, near_m: float, far_m: float, scale: float) -> np.ndarray:
    img = np.vstack([cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), depth_to_color(f.depth, near_m, far_m)])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    draw_boxes_just_outside_their_pixels(img[:img.shape[0] // 2], boxes, scale)
    label = f"{f.camera}  t {seconds_from_stamp(f.stamp):.2f}s  {fps:.0f} fps"
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
                                                 "sim's truth boxes when it sends any. Press q or Esc to quit.")
    parser.add_argument("--address", default=f"tcp://{feed['host']}:{feed['port']}")
    address = parser.parse_args().address
    near_m, far_m = view["depth_display_m"]

    sub = Subscriber([address], [CAMERA_TOPIC_PREFIX, TRUTH_TOPIC], feed["queue_messages"])
    pairer = FramePairer()
    latest: dict[str, tuple[Frame, dict]] = {}
    boxes_of_frames_not_yet_shown: dict[tuple[int, int], dict] = {}
    arrival_times: dict[str, list[float]] = {}
    drawn_stamp = None
    print(f"view: listening on {address}  (q or Esc to quit)")
    cv2.namedWindow(WINDOW)
    cv2.imshow(WINDOW, WAITING_SCREEN)
    while True:
        msg = sub.recv(timeout_ms=50)
        topic, fields, data = msg or (None, None, None)
        f = None
        if topic == TRUTH_TOPIC:
            boxes_of_frames_not_yet_shown[stamp_from_fields(fields["stamp"])] = fields.get("boxes", {})
        elif topic is not None:
            f = pairer.add(topic, fields, data)
        if f is not None:
            boxes_of_frames_not_yet_shown = {s: b for s, b in boxes_of_frames_not_yet_shown.items() if s >= f.stamp}
            latest[f.camera] = f, boxes_of_frames_not_yet_shown.get(f.stamp, {}).get(f.camera, {})
            now = time.perf_counter()
            arrival_times[f.camera] = [t for t in arrival_times.get(f.camera, []) if now - t < 1.0] + [now]
            new_frame_started = f.stamp != drawn_stamp
            if new_frame_started:
                drawn_stamp = f.stamp
                fps_last_second = {name: len(times) for name, times in arrival_times.items()}
                tiles = [tile(*latest[n], fps_last_second[n], near_m, far_m, view["scale"]) for n in sorted(latest)]
                cv2.imshow(WINDOW, compose(tiles))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ESC) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    sub.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
