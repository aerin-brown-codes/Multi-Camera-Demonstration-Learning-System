import numpy as np
import pytest

from feed.pipe import Publisher

WIDTH, HEIGHT = 64, 48
K = np.array([[50.0, 0, 31.5], [0, 51.0, 23.5], [0, 0, 1]])
QUEUE_LARGE_ENOUGH_TO_NEVER_DROP = 200


def images(seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8),
            rng.integers(0, 3000, (HEIGHT, WIDTH), dtype=np.uint16))


def send_until_connected(send, sub):
    for _ in range(100):
        send()
        msg = sub.recv(timeout_ms=20)
        if msg is not None:
            return msg
    raise AssertionError("nothing received")


def everything_waiting(sub, quiet_ms=200):
    received = []
    while (msg := sub.recv(timeout_ms=quiet_ms)) is not None:
        received.append(msg)
    return received


@pytest.fixture
def pub():
    publisher = Publisher("127.0.0.1", "*", QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    yield publisher
    publisher.close()
