import json
from dataclasses import dataclass

import numpy as np
import zmq

from sim.rig import Camera

CAMERA_TOPIC = b"cam/"
TRUTH_TOPIC = b"truth"


@dataclass(frozen=True)
class Frame:
    camera: str
    frame: int
    t: float
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    depth_scale: float


class Publisher:
    def __init__(self, host: str, port, queue_messages: int):
        self._sock = zmq.Context.instance().socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, queue_messages)
        self._sock.bind(f"tcp://{host}:{port}")
        self.address = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)

    def send_frame(self, cam: Camera, frame: int, t: float, rgb: np.ndarray, depth: np.ndarray):
        p = cam.profile
        header = {
            "camera": cam.name, "frame": frame, "t": t, "width": p.width, "height": p.height,
            "K": p.K.tolist(), "depth_scale": p.depth_scale, "rgb_format": "rgb8", "depth_format": "z16",
        }
        self._sock.send_multipart(
            [CAMERA_TOPIC + cam.name.encode(), json.dumps(header).encode(),
             np.ascontiguousarray(rgb), np.ascontiguousarray(depth, dtype="<u2")],
            copy=False,
        )

    def send_truth(self, truth: dict):
        self._sock.send_multipart([TRUTH_TOPIC, json.dumps(truth).encode()])

    def close(self):
        self._sock.close(linger=0)


class _Sub:
    def __init__(self, address: str, topics: list[bytes], queue_messages: int):
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, queue_messages)
        for topic in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, topic)
        self._sock.connect(address)

    def _recv(self, timeout_ms):
        if timeout_ms is not None and not self._sock.poll(timeout_ms):
            return None
        return self._sock.recv_multipart()

    def close(self):
        self._sock.close(linger=0)


class Subscriber(_Sub):
    def __init__(self, address: str = "tcp://127.0.0.1:5555", cameras: list[str] | None = None,
                 queue_messages: int = 10):
        topics = [CAMERA_TOPIC] if cameras is None else [CAMERA_TOPIC + c.encode() for c in cameras]
        super().__init__(address, topics, queue_messages)
        self._cameras = cameras

    def _wants(self, camera: str) -> bool:
        return self._cameras is None or camera in self._cameras

    def recv(self, timeout_ms: int | None = None) -> Frame | None:
        while True:
            parts = self._recv(timeout_ms)
            if parts is None:
                return None
            topic, header, rgb, depth = parts
            h = json.loads(header)
            if self._wants(h["camera"]):
                shape = h["height"], h["width"]
                return Frame(
                    camera=h["camera"], frame=h["frame"], t=h["t"],
                    rgb=np.frombuffer(rgb, np.uint8).reshape(*shape, 3).copy(),
                    depth=np.frombuffer(depth, "<u2").reshape(shape).copy(),
                    K=np.array(h["K"]), depth_scale=h["depth_scale"],
                )

    def __iter__(self):
        while True:
            yield self.recv()


class TruthSubscriber(_Sub):
    def __init__(self, address: str = "tcp://127.0.0.1:5555", queue_messages: int = 10):
        super().__init__(address, [TRUTH_TOPIC], queue_messages)

    def recv(self, timeout_ms: int | None = None) -> dict | None:
        parts = self._recv(timeout_ms)
        return None if parts is None else json.loads(parts[1])

    def __iter__(self):
        while True:
            yield self.recv()
