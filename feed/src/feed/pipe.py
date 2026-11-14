import json

import numpy as np
import zmq


class Publisher:
    def __init__(self, host: str, port, queue_messages: int):
        self._sock = zmq.Context.instance().socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, queue_messages)
        try:
            self._sock.bind(f"tcp://{host}:{port}")
        except zmq.ZMQError as error:
            self._sock.close(linger=0)
            if error.errno != zmq.EADDRINUSE:
                raise
            raise SystemExit(f"port {port} is already in use; is another copy of this program running?") from None
        self.address = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)

    def send(self, topic: str, fields: dict, data: np.ndarray | None = None):
        parts = [topic.encode(), json.dumps(fields).encode()]
        if data is not None:
            parts.append(np.ascontiguousarray(data))
        self._sock.send_multipart(parts)

    def close(self):
        self._sock.close(linger=0)


class Subscriber:
    def __init__(self, addresses: list[str], topic_prefixes: list[str], queue_messages: int):
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, queue_messages)
        for prefix in topic_prefixes:
            self._sock.setsockopt_string(zmq.SUBSCRIBE, prefix)
        for address in addresses:
            self._sock.connect(address)

    def recv(self, timeout_ms: int | None = None) -> tuple[str, dict, bytes | None] | None:
        if timeout_ms is not None and not self._sock.poll(timeout_ms):
            return None
        topic, fields, *data = self._sock.recv_multipart()
        return topic.decode(), json.loads(fields), data[0] if data else None

    def close(self):
        self._sock.close(linger=0)
