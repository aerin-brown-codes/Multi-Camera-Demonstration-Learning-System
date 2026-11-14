import json

import numpy as np
import pytest
from conftest import HEIGHT, WIDTH, K, images

from feed.messages import (METRES_PER_DEPTH_UNIT, K_of, camera_and_kind, camera_info_message, camera_info_topic,
                           camera_topic_prefix, color_message, color_topic, depth_message, depth_topic, image_of,
                           seconds_from_stamp, stamp_fields, stamp_from_fields, stamp_from_seconds, stamp_of)

STAMP = (0, 233333333)
ROS_HEADER_FIELDS = {"stamp", "frame_id"}
ROS_IMAGE_FIELDS_WITHOUT_DATA = {"header", "height", "width", "encoding", "is_bigendian", "step"}
ROS_CAMERA_INFO_FIELDS_WE_SEND = {"header", "height", "width", "distortion_model", "d", "k"}


def over_the_wire(topic, fields, data):
    return topic, json.loads(json.dumps(fields)), None if data is None else data.tobytes()


def test_stamp_is_whole_seconds_and_nanoseconds():
    assert stamp_from_seconds(7 / 30) == (0, 233333333)
    assert stamp_from_seconds(3.0) == (3, 0)
    assert stamp_from_seconds(0.9999999996) == (1, 0), "nanoseconds must never reach 1,000,000,000"
    assert all(type(part) is int for part in stamp_from_seconds(7 / 30))


def test_stamp_survives_the_trip_and_reads_back_as_seconds():
    assert stamp_from_fields(json.loads(json.dumps(stamp_fields(STAMP)))) == STAMP
    assert stamp_fields(STAMP) == {"sec": 0, "nanosec": 233333333}
    assert seconds_from_stamp((3, 500000000)) == 3.5


def test_every_frame_of_a_30_fps_hour_gets_its_own_stamp_in_order():
    stamps = [stamp_from_seconds(frame / 30) for frame in range(30 * 3600)]
    assert stamps == sorted(set(stamps))


def test_topic_names_are_the_realsense_nodes():
    assert camera_info_topic("cam0") == "/camera/cam0/color/camera_info"
    assert color_topic("cam0") == "/camera/cam0/color/image_raw"
    assert depth_topic("cam0") == "/camera/cam0/aligned_depth_to_color/image_raw"
    assert camera_topic_prefix("cam1") == "/camera/cam1/"


def test_camera_name_is_read_back_from_the_topic():
    assert camera_and_kind(color_topic("cam10")) == ("cam10", "color/image_raw")
    assert camera_and_kind(depth_topic("cam10"))[0] == "cam10"
    assert camera_and_kind(camera_info_topic("cam10"))[0] == "cam10"


@pytest.mark.parametrize("topic", ["truth", "/detect/cam0/detections", "/camera/cam0/infra1/image_raw", "/camera//color/image_raw"])
def test_a_topic_that_is_not_one_of_a_cameras_three_is_refused_by_name(topic):
    with pytest.raises(ValueError, match=topic):
        camera_and_kind(topic)


def test_colour_and_depth_survive_the_trip_unchanged_and_writable():
    rgb, depth = images(0)
    rgb_back = image_of(*over_the_wire(*color_message("cam0", STAMP, rgb)))
    depth_back = image_of(*over_the_wire(*depth_message("cam0", STAMP, depth)))
    assert np.array_equal(rgb_back, rgb) and rgb_back.dtype == np.uint8
    assert np.array_equal(depth_back, depth) and depth_back.dtype == np.uint16
    assert rgb_back.flags.writeable and depth_back.flags.writeable


def test_image_fields_are_exactly_ros_sensor_msgs_image():
    rgb, depth = images(0)
    _, color_fields, _ = color_message("cam0", STAMP, rgb)
    _, depth_fields, _ = depth_message("cam0", STAMP, depth)
    for fields in (color_fields, depth_fields):
        assert set(fields) == ROS_IMAGE_FIELDS_WITHOUT_DATA and set(fields["header"]) == ROS_HEADER_FIELDS
        assert (fields["height"], fields["width"], fields["is_bigendian"]) == (HEIGHT, WIDTH, 0)
        assert fields["header"]["frame_id"] == "cam0_color_optical_frame", "aligned depth carries the colour frame"
        assert stamp_of(fields) == STAMP
    assert (color_fields["encoding"], color_fields["step"]) == ("rgb8", WIDTH * 3)
    assert (depth_fields["encoding"], depth_fields["step"]) == ("16UC1", WIDTH * 2)


def test_camera_info_carries_the_lens_matrix_and_no_distortion():
    topic, fields, data = camera_info_message("cam0", STAMP, K, WIDTH, HEIGHT)
    assert data is None and set(fields) == ROS_CAMERA_INFO_FIELDS_WE_SEND
    _, fields, _ = over_the_wire(topic, fields, data)
    assert np.array_equal(K_of(fields), K) and K_of(fields).shape == (3, 3)
    assert fields["d"] == [0.0] * 5 and fields["distortion_model"] == "plumb_bob"
    assert (fields["height"], fields["width"], stamp_of(fields)) == (HEIGHT, WIDTH, STAMP)


def test_depth_unit_is_the_ros_millimetre():
    assert METRES_PER_DEPTH_UNIT == 0.001


@pytest.mark.parametrize("change, complaint", [
    ({"encoding": "bgr8"}, "bgr8"),
    ({"is_bigendian": 1}, "big-endian"),
    ({"step": WIDTH * 4}, "step"),
    ({"height": HEIGHT + 1}, "bytes"),
])
def test_a_colour_image_that_is_not_what_its_topic_carries_is_refused_by_topic(change, complaint):
    topic, fields, data = over_the_wire(*color_message("cam0", STAMP, images(0)[0]))
    with pytest.raises(ValueError, match=f"{topic}.*{complaint}"):
        image_of(topic, {**fields, **change}, data)


def test_an_image_with_no_pixels_or_too_few_is_refused_by_topic():
    topic, fields, data = over_the_wire(*depth_message("cam0", STAMP, images(0)[1]))
    for pixels in (None, data[:-1]):
        with pytest.raises(ValueError, match=f"{topic}.*bytes"):
            image_of(topic, fields, pixels)


def test_camera_info_is_not_read_as_an_image():
    with pytest.raises(ValueError, match="camera info"):
        image_of(*camera_info_message("cam0", STAMP, K, WIDTH, HEIGHT))


@pytest.mark.parametrize("wrong_depth", [
    np.full((HEIGHT, WIDTH), 0.86),
    np.zeros((HEIGHT, WIDTH), ">u2"),
    np.zeros((HEIGHT, WIDTH, 1), np.uint16),
])
def test_depth_that_is_not_16_bit_millimetres_is_refused_before_it_is_sent(wrong_depth):
    with pytest.raises(ValueError, match="16UC1"):
        depth_message("cam0", STAMP, wrong_depth)


def test_colour_that_is_not_three_bytes_a_pixel_is_refused_before_it_is_sent():
    for wrong_colour in (np.zeros((HEIGHT, WIDTH), np.uint8), np.zeros((HEIGHT, WIDTH, 4), np.uint8),
                         np.zeros((HEIGHT, WIDTH, 3), np.float32)):
        with pytest.raises(ValueError, match="rgb8"):
            color_message("cam0", STAMP, wrong_colour)
