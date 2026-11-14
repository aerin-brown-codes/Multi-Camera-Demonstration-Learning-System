from dataclasses import dataclass

import numpy as np

from feed.messages import (CAMERA_INFO, COLOR, DEPTH, K_of, camera_and_kind, camera_info_message, color_message,
                           depth_message, image_of, stamp_of)
from feed.pipe import Publisher


@dataclass(frozen=True)
class Frame:
    camera: str
    stamp: tuple[int, int]
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray


def send_frame(publisher: Publisher, camera: str, stamp: tuple[int, int], K: np.ndarray,
               rgb: np.ndarray, depth: np.ndarray):
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"{camera}: depth must be aligned to colour, but colour is {rgb.shape[1]}x{rgb.shape[0]} "
                         f"and depth is {depth.shape[1]}x{depth.shape[0]}")
    height, width = depth.shape
    messages = [camera_info_message(camera, stamp, K, width, height),
                color_message(camera, stamp, rgb), depth_message(camera, stamp, depth)]
    for message in messages:
        publisher.send(*message)


class FramePairer:
    def __init__(self):
        self._K: dict[str, np.ndarray] = {}
        self._newest: dict[str, dict[str, tuple[tuple[int, int], np.ndarray]]] = {}

    def add(self, topic: str, fields: dict, data: bytes | None) -> Frame | None:
        camera, kind = camera_and_kind(topic)
        newest = self._newest.setdefault(camera, {})
        if kind == CAMERA_INFO:
            self._K[camera] = K_of(fields)
        else:
            newest[kind] = stamp_of(fields), image_of(topic, fields, data)
        if camera not in self._K or COLOR not in newest or DEPTH not in newest:
            return None
        (color_stamp, rgb), (depth_stamp, depth) = newest[COLOR], newest[DEPTH]
        if color_stamp != depth_stamp:
            return None
        newest.clear()
        return Frame(camera, color_stamp, rgb, depth, self._K[camera])
