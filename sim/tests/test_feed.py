import numpy as np

from sim.feed import Publisher, Subscriber, TruthSubscriber
from sim.rig import Camera, Profile, look_at

PROFILE = Profile(width=64, height=48, fx=50.0, fy=51.0, cx=31.5, cy=23.5,
                  depth_min_m=0.2, depth_max_m=3.0, depth_scale=0.001)
QUEUE_LARGE_ENOUGH_TO_NEVER_DROP = 100


def camera(name):
    return Camera(name, PROFILE, np.array([1.0, 0, 1]), look_at([1.0, 0, 1], [0, 0, 0]))


def images(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), rng.integers(0, 3000, (48, 64), dtype=np.uint16)


def send_until_connected(send, sub):
    for _ in range(100):
        send()
        msg = sub.recv(timeout_ms=20)
        if msg is not None:
            return msg
    raise AssertionError("nothing received")


def test_frame_roundtrip():
    pub = Publisher("127.0.0.1", "*", queue_messages=10)
    sub = Subscriber(pub.address)
    rgb, depth = images(0)
    f = send_until_connected(lambda: pub.send_frame(camera("cam0"), 7, 0.25, rgb, depth), sub)
    assert (f.camera, f.frame, f.t, f.depth_scale) == ("cam0", 7, 0.25, 0.001)
    assert np.array_equal(f.rgb, rgb) and np.array_equal(f.depth, depth)
    assert f.depth.dtype == np.uint16 and f.rgb.flags.writeable
    assert np.allclose(f.K, PROFILE.K)
    sub.close()
    pub.close()


def test_camera_filter_ignores_names_sharing_a_prefix():
    pub = Publisher("127.0.0.1", "*", queue_messages=QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    sub = Subscriber(pub.address, cameras=["cam1"], queue_messages=QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    rgb, depth = images(1)

    def send_cam0_cam10_cam1():
        for name in ["cam0", "cam10", "cam1"]:
            pub.send_frame(camera(name), 0, 0.0, rgb, depth)

    send_until_connected(send_cam0_cam10_cam1, sub)
    for _ in range(5):
        send_cam0_cam10_cam1()
    received = []
    while (f := sub.recv(timeout_ms=200)) is not None:
        received.append(f.camera)
    assert len(received) >= 5 and set(received) == {"cam1"}
    sub.close()
    pub.close()


def test_truth_never_reaches_frame_subscribers():
    pub = Publisher("127.0.0.1", "*", queue_messages=10)
    frames, truth = Subscriber(pub.address), TruthSubscriber(pub.address)
    msg = send_until_connected(lambda: pub.send_truth({"frame": 3, "t": 0.1}), truth)
    assert msg == {"frame": 3, "t": 0.1}
    assert frames.recv(timeout_ms=100) is None
    frames.close()
    truth.close()
    pub.close()
