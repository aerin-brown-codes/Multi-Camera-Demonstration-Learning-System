import json

import pytest

from feed.detections import detections_message, detections_of

STAMP = (0, 233333333)
FRAME_ID = "cam0_color_optical_frame"
FOUND = [("cube", (100.0, 80.0, 160.0, 130.0), 0.91), ("hand", (0.25, 3.5, 847.75, 479.125), 0.2501)]


def over_the_wire(topic, fields, data):
    assert data is None, "a boxes message has no pixels"
    return topic, json.loads(json.dumps(fields))


def test_boxes_read_back_are_the_classes_corner_boxes_and_confidences_that_were_written():
    camera, stamp, found = detections_of(*over_the_wire(*detections_message("cam0", STAMP, FRAME_ID, FOUND)))
    assert (camera, stamp) == ("cam0", STAMP)
    assert [(class_name, confidence) for class_name, _, confidence in found] == [("cube", 0.91), ("hand", 0.2501)]
    for (_, box, _), (_, written, _) in zip(found, FOUND, strict=True):
        assert box == pytest.approx(written, abs=1e-9)


def test_a_frame_with_nothing_found_is_still_a_message_with_its_stamp():
    topic, fields = over_the_wire(*detections_message("cam3", STAMP, "cam3_color_optical_frame", []))
    assert fields["detections"] == []
    assert detections_of(topic, fields) == ("cam3", STAMP, [])


def test_the_message_is_ros_detection2darray_field_for_field():
    topic, fields = over_the_wire(*detections_message("cam0", STAMP, FRAME_ID, FOUND[:1]))
    assert topic == "/detect/cam0/detections"
    assert fields == {
        "header": {"stamp": {"sec": 0, "nanosec": 233333333}, "frame_id": "cam0_color_optical_frame"},
        "detections": [{
            "results": [{"hypothesis": {"class_id": "cube", "score": 0.91}}],
            "bbox": {"center": {"position": {"x": 130.0, "y": 105.0}, "theta": 0.0}, "size_x": 60.0, "size_y": 50.0},
        }]}


@pytest.mark.parametrize("topic", ["truth", "/camera/cam0/color/image_raw", "/detect/cam0", "/detect//detections",
                                   "/detect/cam0/other"])
def test_a_topic_that_is_not_a_cameras_detections_is_refused_by_name(topic):
    with pytest.raises(ValueError, match=topic):
        detections_of(topic, {"header": {"stamp": {"sec": 0, "nanosec": 0}}, "detections": []})
