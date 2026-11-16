from dataclasses import replace

import numpy as np
import pytest
from feed.frames import FramePairer
from feed.messages import CAMERA_TOPIC_PREFIX, TRUTH_TOPIC, stamp_from_fields, stamp_from_seconds, stamp_of
from feed.pipe import Publisher, Subscriber
from feed.transforms import TABLE_FRAME, TF_STATIC_TOPIC, static_transforms_of

from sim.rig import Rig
from sim.run import (held_pose_or_none_without_poses, make_truth, require_depth_in_the_feeds_unit,
                     send_truth_then_camera_poses_then_every_camera)

QUEUE_LARGE_ENOUGH_TO_NEVER_DROP = 200
STAMP = stamp_from_seconds(7 / 30)


def connected_pair():
    pub = Publisher("127.0.0.1", "*", QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    sub = Subscriber([pub.address], [CAMERA_TOPIC_PREFIX, TRUTH_TOPIC, TF_STATIC_TOPIC], QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    for _ in range(100):
        pub.send(TRUTH_TOPIC, {"warm_up": True})
        if sub.recv(timeout_ms=20) is not None:
            break
    else:
        raise AssertionError("subscriber never connected")
    while sub.recv(timeout_ms=200) is not None:
        pass
    return pub, sub


def test_one_frame_arrives_as_truth_then_camera_poses_then_each_cameras_three_messages_with_one_stamp(tabletop):
    s = tabletop
    pub, sub = connected_pair()
    truth = make_truth(s.model, s.data, s.rig, 7)
    send_truth_then_camera_poses_then_every_camera(pub, s.rig, truth, STAMP, s.frames)
    received = [sub.recv(timeout_ms=500) for _ in range(2 + 3 * len(s.rig.cameras))]
    assert sub.recv(timeout_ms=100) is None
    sub.close()
    pub.close()

    assert [topic for topic, _, _ in received] == ["truth", "/tf_static"] + [
        f"/camera/{cam.name}/{kind}" for cam in s.rig.cameras
        for kind in ["color/camera_info", "color/image_raw", "aligned_depth_to_color/image_raw"]]
    _, truth_received, _ = received[0]
    assert stamp_from_fields(truth_received["stamp"]) == STAMP
    assert (truth_received["frame"], truth_received["t"]) == (7, truth["t"]), "truth keeps frame and t"
    assert truth_received["bodies"] == truth["bodies"] and "stamp" not in truth, "the caller's truth is left alone"
    assert {stamp_of(fields) for _, fields, _ in received[2:]} == {STAMP}

    _, poses_received, _ = received[1]
    assert {stamp_of(pose) for pose in poses_received["transforms"]} == {STAMP}
    T_table_cam_of = static_transforms_of(poses_received, TABLE_FRAME)
    assert list(T_table_cam_of) == [f"{cam.name}_color_optical_frame" for cam in s.rig.cameras]
    for cam in s.rig.cameras:
        off = np.abs(T_table_cam_of[f"{cam.name}_color_optical_frame"] - cam.T_world_cam).max()
        assert off < 1e-12, f"{cam.name}'s pose on the feed is not the rig's"

    pairer = FramePairer()
    frames = [f for f in (pairer.add(*message) for message in received[2:]) if f is not None]
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


def test_truth_names_the_pose_that_is_held_and_how_many_there_are(tabletop):
    names = ["cube_alone", "hand_over_cube", "hands_apart"]
    held = [held_pose_or_none_without_poses(frame, fps=30, hold_s=3.0, names=names) for frame in (0, 89, 90, 270)]
    assert held == [{"index": 0, "name": "cube_alone", "count": 3}, {"index": 0, "name": "cube_alone", "count": 3},
                    {"index": 1, "name": "hand_over_cube", "count": 3}, {"index": 0, "name": "cube_alone", "count": 3}]
    truth = make_truth(tabletop.model, tabletop.data, tabletop.rig, 90, held_pose=held[2])
    assert truth["pose"] == {"index": 1, "name": "hand_over_cube", "count": 3}


def test_a_scene_without_poses_sends_no_pose_in_truth(tabletop):
    assert held_pose_or_none_without_poses(7, fps=30, hold_s=3.0, names=[]) is None
    assert "pose" not in make_truth(tabletop.model, tabletop.data, tabletop.rig, 7)
