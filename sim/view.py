import argparse
import math
import time

import cv2
import numpy as np
import yaml

from sim.feed import Frame, Subscriber
from sim.run import DEFAULT_CONFIG

WINDOW = "camera feed"
WAITING_SCREEN = np.zeros((240, 480, 3), np.uint8)
ESC = 27


def depth_to_color(depth: np.ndarray, depth_scale: float, near_m: float, far_m: float) -> np.ndarray:
    metres = depth * depth_scale
    nearness = np.clip((far_m - metres) / (far_m - near_m), 0, 1)
    colored = cv2.applyColorMap((nearness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth == 0] = 0
    return colored


def tile(f: Frame, fps: float, near_m: float, far_m: float, scale: float) -> np.ndarray:
    img = np.vstack([cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), depth_to_color(f.depth, f.depth_scale, near_m, far_m)])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    label = f"{f.camera}  frame {f.frame}  t {f.t:.2f}s  {fps:.0f} fps"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), cv2.FILLED)
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
    parser = argparse.ArgumentParser(description="Show every camera in the feed live, color above depth. "
                                                 "Press q or Esc to quit.")
    parser.add_argument("--address", default=f"tcp://{feed['host']}:{feed['port']}")
    address = parser.parse_args().address
    near_m, far_m = view["depth_display_m"]

    sub = Subscriber(address, queue_messages=feed["queue_messages"])
    latest: dict[str, Frame] = {}
    arrival_times: dict[str, list[float]] = {}
    drawn_frame = -1
    print(f"view: listening on {address}  (q or Esc to quit)")
    cv2.namedWindow(WINDOW)
    cv2.imshow(WINDOW, WAITING_SCREEN)
    while True:
        f = sub.recv(timeout_ms=50)
        if f is not None:
            latest[f.camera] = f
            now = time.perf_counter()
            arrival_times[f.camera] = [t for t in arrival_times.get(f.camera, []) if now - t < 1.0] + [now]
            new_frame_started = f.frame != drawn_frame
            if new_frame_started:
                drawn_frame = f.frame
                fps_last_second = {name: len(times) for name, times in arrival_times.items()}
                tiles = [tile(latest[n], fps_last_second[n], near_m, far_m, view["scale"]) for n in sorted(latest)]
                cv2.imshow(WINDOW, compose(tiles))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ESC) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    sub.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
