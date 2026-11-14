from dataclasses import replace

import numpy as np
import pytest
from feed.frames import FramePairer
from feed.messages import CAMERA_TOPIC_PREFIX, TRUTH_TOPIC, stamp_from_fields, stamp_from_seconds, stamp_of
from feed.pipe import Publisher, Subscriber

from sim.rig import Rig
from sim.run import make_truth, require_depth_in_the_feeds_unit, send_truth_then_every_camera

QUEUE_LARGE_ENOUGH_TO_NEVER_DROP = 200
STAMP = stamp_from_seconds(7 / 30)


def connected_pair():
    pub = Publisher("127.0.0.1", "*", QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    sub = Subscriber([pub.address], [CAMERA_TOPIC_PREFIX, TRUTH_TOPIC], QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    for _ in range(100):
        pub.send(TRUTH_TOPIC, {"warm_up": True})
        if sub.recv(timeout_ms=20) is not None:
            break
    else:
        raise AssertionError("subscriber never connected")
    while sub.recv(timeout_ms=200) is not None:
        pass
    return pub, sub


def test_one_frame_arrives_as_truth_then_each_cameras_three_messages_with_one_stamp_and_pairs_back(tabletop):
    s = tabletop
    pub, sub = connected_pair()
    truth = make_truth(s.model, s.data, s.rig, 7)
    send_truth_then_every_camera(pub, s.rig, truth, STAMP, s.frames)
    received = [sub.recv(timeout_ms=500) for _ in range(1 + 3 * len(s.rig.cameras))]
    assert sub.recv(timeout_ms=100) is None
    sub.close()
    pub.close()

    assert [topic for topic, _, _ in received] == ["truth"] + [
        f"/camera/{cam.name}/{kind}" for cam in s.rig.cameras
        for kind in ["color/camera_info", "color/image_raw", "aligned_depth_to_color/image_raw"]]
    _, truth_received, _ = received[0]
    assert stamp_from_fields(truth_received["stamp"]) == STAMP
    assert (truth_received["frame"], truth_received["t"]) == (7, truth["t"]), "truth keeps frame and t"
    assert truth_received["bodies"] == truth["bodies"] and "stamp" not in truth, "the caller's truth is left alone"
    assert {stamp_of(fields) for _, fields, _ in received[1:]} == {STAMP}

    pairer = FramePairer()
    frames = [f for f in (pairer.add(*message) for message in received[1:]) if f is not None]
    assert [f.camera for f in frames] == [cam.name for cam in s.rig.cameras]
    for f, cam in zip(frames, s.rig.cameras):
        rgb, depth = s.frames[cam.name]
        assert f.stamp == STAMP and np.array_equal(f.K, cam.profile.K)
        assert np.array_equal(f.rgb, rgb) and np.array_equal(f.depth, depth), f"{cam.name} is not what was rendered"


def test_the_rig_as_configured_has_depth_in_the_feeds_unit(tabletop):
    require_depth_in_the_feeds_unit(tabletop.rig)


def test_a_profile_with_another_depth_scale_is_refused_by_camera_name(tabletop):
    cameras = list(tabletop.rig.cameras)
    cameras[2] = replace(cameras[2], profile=replace(cameras[2].profile, depth_scale=0.0001))
    with pytest.raises(ValueError, match=f"{cameras[2].name}.*0.0001"):
        require_depth_in_the_feeds_unit(Rig(tabletop.rig.fps, cameras))
