import json

import numpy as np
import pytest
from conftest import HEIGHT, WIDTH, K, images

from feed.frames import FramePairer, send_frame
from feed.messages import camera_info_message, color_message, depth_message

FIRST, SECOND = (0, 233333333), (0, 266666667)


def over_the_wire(topic, fields, data):
    return topic, json.loads(json.dumps(fields)), None if data is None else data.tobytes()


def info(camera, stamp):
    return over_the_wire(*camera_info_message(camera, stamp, K, WIDTH, HEIGHT))


def color(camera, stamp, seed=0):
    return over_the_wire(*color_message(camera, stamp, images(seed)[0]))


def depth(camera, stamp, seed=0):
    return over_the_wire(*depth_message(camera, stamp, images(seed)[1]))


def frames_from(messages):
    pairer = FramePairer()
    return [frame for frame in (pairer.add(*message) for message in messages) if frame is not None]


class RecordingPublisher:
    def __init__(self):
        self.sent = []

    def send(self, topic, fields, data=None):
        self.sent.append(over_the_wire(topic, fields, data))


def test_info_then_colour_then_depth_gives_the_frame_that_was_sent():
    rgb, depth_image = images(0)
    pairer = FramePairer()
    assert pairer.add(*info("cam0", FIRST)) is None
    assert pairer.add(*color("cam0", FIRST)) is None
    frame = pairer.add(*depth("cam0", FIRST))
    assert (frame.camera, frame.stamp) == ("cam0", FIRST)
    assert np.array_equal(frame.rgb, rgb) and np.array_equal(frame.depth, depth_image)
    assert frame.depth.dtype == np.uint16 and np.array_equal(frame.K, K)


def test_depth_before_colour_gives_a_frame():
    assert [f.stamp for f in frames_from([info("cam0", FIRST), depth("cam0", FIRST), color("cam0", FIRST)])] == [FIRST]


def test_a_colour_image_whose_depth_never_comes_never_pairs_with_a_later_depth():
    frames = frames_from([info("cam0", FIRST), color("cam0", FIRST, seed=1),
                          depth("cam0", SECOND, seed=2), color("cam0", SECOND, seed=2)])
    assert [f.stamp for f in frames] == [SECOND]
    assert np.array_equal(frames[0].rgb, images(2)[0]), "the frame must hold the colour of its own stamp"


def test_a_depth_image_whose_colour_never_comes_never_pairs_with_a_later_colour():
    frames = frames_from([info("cam0", FIRST), depth("cam0", FIRST, seed=1),
                          color("cam0", SECOND, seed=2), depth("cam0", SECOND, seed=2)])
    assert [f.stamp for f in frames] == [SECOND]
    assert np.array_equal(frames[0].depth, images(2)[1])


def test_no_frame_before_a_camera_info_and_one_as_soon_as_it_comes():
    pairer = FramePairer()
    assert pairer.add(*color("cam0", FIRST)) is None
    assert pairer.add(*depth("cam0", FIRST)) is None
    assert pairer.add(*info("cam0", SECOND)).stamp == FIRST


def test_the_newest_camera_info_is_used_whatever_its_stamp():
    assert len(frames_from([info("cam0", FIRST), color("cam0", SECOND), depth("cam0", SECOND)])) == 1


def test_a_frame_is_given_out_once():
    frames = frames_from([info("cam0", FIRST), color("cam0", FIRST), depth("cam0", FIRST), info("cam0", SECOND),
                          depth("cam0", FIRST)])
    assert len(frames) == 1


def test_two_cameras_whose_messages_are_mixed_pair_separately():
    frames = frames_from([info("cam0", FIRST), info("cam1", FIRST), color("cam0", FIRST, seed=0),
                          color("cam1", FIRST, seed=1), depth("cam1", FIRST, seed=1), depth("cam0", FIRST, seed=0)])
    assert [f.camera for f in frames] == ["cam1", "cam0"]
    for frame, seed in zip(frames, (1, 0)):
        assert np.array_equal(frame.rgb, images(seed)[0]) and np.array_equal(frame.depth, images(seed)[1])


def test_one_cameras_info_does_not_serve_another():
    assert frames_from([info("cam0", FIRST), color("cam1", FIRST), depth("cam1", FIRST)]) == []


@pytest.mark.parametrize("topic", ["truth", "/detect/cam0/detections"])
def test_the_pairer_refuses_a_topic_that_is_not_a_cameras(topic):
    with pytest.raises(ValueError, match=topic):
        FramePairer().add(topic, {}, None)


def test_send_frame_sends_info_then_colour_then_depth_with_one_stamp_and_they_pair_back():
    rgb, depth_image = images(3)
    publisher = RecordingPublisher()
    send_frame(publisher, "cam2", FIRST, K, rgb, depth_image)
    assert [topic for topic, _, _ in publisher.sent] == [
        "/camera/cam2/color/camera_info", "/camera/cam2/color/image_raw",
        "/camera/cam2/aligned_depth_to_color/image_raw"]
    assert {(f["header"]["stamp"]["sec"], f["header"]["stamp"]["nanosec"]) for _, f, _ in publisher.sent} == {FIRST}
    assert (publisher.sent[0][1]["width"], publisher.sent[0][1]["height"]) == (WIDTH, HEIGHT)
    (frame,) = frames_from(publisher.sent)
    assert np.array_equal(frame.rgb, rgb) and np.array_equal(frame.depth, depth_image) and np.array_equal(frame.K, K)


def test_send_frame_refuses_depth_that_is_not_the_colour_images_size_and_sends_nothing():
    rgb, _ = images(0)
    publisher = RecordingPublisher()
    with pytest.raises(ValueError, match="cam0.*aligned"):
        send_frame(publisher, "cam0", FIRST, K, rgb, np.zeros((HEIGHT // 2, WIDTH // 2), np.uint16))
    assert publisher.sent == []
