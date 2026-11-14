import numpy as np
import pytest
import zmq
from conftest import QUEUE_LARGE_ENOUGH_TO_NEVER_DROP, K, everything_waiting, images, send_until_connected

from feed.frames import send_frame
from feed.messages import CAMERA_TOPIC_PREFIX, TRUTH_TOPIC, camera_topic_prefix, color_topic
from feed.pipe import Publisher, Subscriber

STAMP = (0, 0)


def subscriber(pub, topic_prefixes):
    return Subscriber([pub.address], topic_prefixes, QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)


def test_fields_and_pixels_arrive_as_sent(pub):
    sub = subscriber(pub, ["/camera/"])
    rgb, _ = images(0)
    topic, fields, data = send_until_connected(lambda: pub.send(color_topic("cam0"), {"a": [1, 2.5, "x"]}, rgb), sub)
    assert (topic, fields) == ("/camera/cam0/color/image_raw", {"a": [1, 2.5, "x"]})
    assert type(data) is bytes and data == rgb.tobytes()
    sub.close()


def test_a_message_without_pixels_arrives_with_none(pub):
    sub = subscriber(pub, [TRUTH_TOPIC])
    assert send_until_connected(lambda: pub.send(TRUTH_TOPIC, {"frame": 3}), sub) == ("truth", {"frame": 3}, None)
    sub.close()


def test_pixels_are_sent_as_they_were_at_the_call_even_if_the_array_changes_after(pub):
    sub = subscriber(pub, ["/camera/"])
    send_until_connected(lambda: pub.send("/camera/warm_up", {}), sub)
    everything_waiting(sub)
    rgb, _ = images(1)
    as_sent = rgb.tobytes()
    pub.send(color_topic("cam0"), {}, rgb)
    rgb[:] = 0
    assert sub.recv(timeout_ms=500)[2] == as_sent
    sub.close()


def test_a_subscriber_to_cam1_gets_nothing_of_cam10(pub):
    sub = subscriber(pub, [camera_topic_prefix("cam1")])
    rgb, depth = images(1)

    def send_cam0_cam10_cam1():
        for name in ["cam0", "cam10", "cam1"]:
            send_frame(pub, name, STAMP, K, rgb, depth)

    send_until_connected(send_cam0_cam10_cam1, sub)
    for _ in range(5):
        send_cam0_cam10_cam1()
    topics = [topic for topic, _, _ in everything_waiting(sub)]
    assert len(topics) >= 15 and all(topic.startswith("/camera/cam1/") for topic in topics)
    sub.close()


def test_truth_never_reaches_a_camera_subscriber(pub):
    cameras, truth = subscriber(pub, [CAMERA_TOPIC_PREFIX]), subscriber(pub, [TRUTH_TOPIC])
    send_until_connected(lambda: pub.send(TRUTH_TOPIC, {"frame": 3}), truth)
    assert cameras.recv(timeout_ms=100) is None
    cameras.close()
    truth.close()


def test_truth_sent_before_its_frames_is_read_before_them_on_one_connection(pub):
    sub = subscriber(pub, [CAMERA_TOPIC_PREFIX, TRUTH_TOPIC])
    rgb, depth = images(2)
    send_until_connected(lambda: pub.send(TRUTH_TOPIC, {"frame": -1}), sub)
    everything_waiting(sub)
    for frame in range(3):
        pub.send(TRUTH_TOPIC, {"frame": frame})
        for name in ["cam0", "cam1"]:
            send_frame(pub, name, (frame, 0), K, rgb, depth)
    received = [topic for topic, _, _ in everything_waiting(sub)]
    one_frame = ["truth"] + [f"/camera/{name}/{kind}" for name in ["cam0", "cam1"] for kind in
                             ["color/camera_info", "color/image_raw", "aligned_depth_to_color/image_raw"]]
    assert received == one_frame * 3
    sub.close()


def test_one_subscriber_reads_two_publishers(pub):
    other = Publisher("127.0.0.1", "*", QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    sub = Subscriber([pub.address, other.address], ["/camera/", "/detect/"], QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)

    def send_on_both():
        pub.send("/camera/cam0/color/camera_info", {})
        other.send("/detect/cam0/detections", {})

    send_until_connected(send_on_both, sub)
    for _ in range(20):
        send_on_both()
    assert {topic for topic, _, _ in everything_waiting(sub)} == {"/camera/cam0/color/camera_info",
                                                                 "/detect/cam0/detections"}
    sub.close()
    other.close()


def test_a_second_publisher_on_a_taken_port_stops_with_the_port_in_its_message(pub):
    port = pub.address.rsplit(":", 1)[1]
    with pytest.raises(SystemExit, match=f"port {port} is already in use"):
        Publisher("127.0.0.1", port, 10)


def test_a_host_that_cannot_be_bound_keeps_its_own_error():
    with pytest.raises(zmq.ZMQError):
        Publisher("no.such.host.invalid", "*", 10)
