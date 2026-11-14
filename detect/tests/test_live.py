import sys

import numpy as np
import pytest
from feed.detections import DETECTIONS_TOPIC_PREFIX, detections_of
from feed.frames import send_frame
from feed.messages import TRUTH_TOPIC, camera_info_message, color_message, depth_message
from feed.pipe import Publisher, Subscriber

from detect import detector, live
from detect.scoring import Detection

WIDTH, HEIGHT = 64, 48
K = np.eye(3)
QUEUE_LARGE_ENOUGH_TO_NEVER_DROP = 200


def colour_filled_with(value):
    return np.full((HEIGHT, WIDTH, 3), value, np.uint8)


DEPTH = np.zeros((HEIGHT, WIDTH), np.uint16)


def as_received(topic, fields, data):
    return topic, fields, None if data is None else data.tobytes()


def colour(camera, stamp, value):
    return as_received(*color_message(camera, stamp, colour_filled_with(value)))


def a_box_as_wide_as_the_images_value(rgb_images):
    return [[Detection("cube", (0.0, 0.0, float(rgb[0, 0, 0]), 10.0), 0.5)] for rgb in rgb_images]


class SubscriberWithThese:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv(self, timeout_ms=None):
        return self.messages.pop(0) if self.messages else None


class RecordingPublisher:
    def __init__(self):
        self.sent = []

    def send(self, topic, fields, data=None):
        self.sent.append((topic, fields, data))


@pytest.fixture
def live_cfg(cfg, run_dir):
    return {"runs_dir": cfg["runs_dir"], "run": run_dir.name, "device": "cpu",
            "feed_address": "tcp://127.0.0.1:5555", "publish": {"host": "127.0.0.1", "port": "*"},
            "queue_messages": QUEUE_LARGE_ENOUGH_TO_NEVER_DROP, "report_every_s": 5}


@pytest.fixture
def nothing_slow_may_start(monkeypatch):
    def must_not_be_reached(*args, **kwargs):
        raise AssertionError("the program must have stopped before this")

    monkeypatch.setattr(live, "Publisher", must_not_be_reached)
    return must_not_be_reached


def test_a_run_that_does_not_exist_stops_before_port_and_model_and_lists_the_runs_there_are(
        live_cfg, run_dir, nothing_slow_may_start):
    (run_dir.parent / "second").mkdir()
    (run_dir.parent / "a.log").touch()
    with pytest.raises(SystemExit) as stopped:
        live.started(live_cfg, "run9", load_detector=nothing_slow_may_start)
    assert str(run_dir.parent / "run9") in str(stopped.value)
    assert str(stopped.value).endswith("first, second"), "the runs that exist are listed, files beside them are not"


@pytest.mark.parametrize("missing, what_to_do", [
    ("weights/best.pt", "colab.sh fetch first"),
    ("train.yaml", "colab.sh fetch first"),
    ("cut_off.json", "detect.score --run"),
])
def test_a_run_missing_a_file_stops_before_port_and_model_with_the_path_and_what_to_do(
        live_cfg, run_dir, nothing_slow_may_start, missing, what_to_do):
    (run_dir / missing).unlink()
    with pytest.raises(SystemExit) as stopped:
        live.started(live_cfg, run_dir.name, load_detector=nothing_slow_may_start)
    assert str(run_dir / missing) in str(stopped.value) and what_to_do in str(stopped.value)


def test_a_taken_port_stops_it_before_the_model_is_loaded(live_cfg, run_dir):
    def must_not_be_reached(*args):
        raise AssertionError("the model must not be loaded when the port is taken")

    first = Publisher("127.0.0.1", "*", 10)
    port = first.address.rsplit(":", 1)[1]
    with pytest.raises(SystemExit, match=f"port {port} is already in use"):
        live.started({**live_cfg, "publish": {"host": "127.0.0.1", "port": port}}, run_dir.name, must_not_be_reached)
    first.close()


def test_weights_with_other_class_names_stop_it_before_the_feed_and_free_its_port(live_cfg, run_dir, monkeypatch):
    class WeightsForOtherClasses:
        names = {0: "cube", 1: "hand"}

    def must_not_be_reached(*args):
        raise AssertionError("the feed must not be connected to when the weights are refused")

    monkeypatch.setattr(detector, "YOLO", lambda weights: WeightsForOtherClasses())
    monkeypatch.setattr(live, "Subscriber", must_not_be_reached)
    port = Publisher("127.0.0.1", "*", 10)
    free_port = port.address.rsplit(":", 1)[1]
    port.close()
    with pytest.raises(ValueError, match=r"trained for \['cube', 'hand'\].*\['hand', 'cube'\]"):
        live.started({**live_cfg, "publish": {"host": "127.0.0.1", "port": free_port}}, run_dir.name)
    Publisher("127.0.0.1", free_port, 10).close()


def test_run_on_the_command_line_is_used_instead_of_the_configs(monkeypatch):
    asked_for = []

    def note_the_run_and_stop(cfg, run):
        asked_for.append((cfg["run"], run))
        raise SystemExit("noted")

    monkeypatch.setattr(live, "started", note_the_run_and_stop)
    for arguments in ([], ["--run", "run2"]):
        monkeypatch.setattr(sys, "argv", ["detect.live", *arguments])
        with pytest.raises(SystemExit, match="noted"):
            live.main()
    assert asked_for == [("run1", "run1"), ("run1", "run2")]


def test_of_several_waiting_colour_images_the_newest_of_each_camera_is_used_and_the_others_are_counted():
    first, second, third = (0, 1), (0, 2), (0, 3)
    newest, skipped = live.newest_colour_image_of_each_camera([
        as_received(TRUTH_TOPIC, {"frame": 0}, None),
        as_received(*camera_info_message("cam0", first, K, WIDTH, HEIGHT)),
        colour("cam0", first, 1), as_received(*depth_message("cam0", first, DEPTH)),
        colour("cam1", first, 1), colour("cam0", second, 2), colour("cam0", third, 3),
        as_received(*depth_message("cam1", third, DEPTH)),
    ])
    assert {camera: fields["header"]["stamp"]["nanosec"] for camera, (_, fields, _) in newest.items()} == {
        "cam0": 3, "cam1": 1}
    assert dict(skipped) == {"cam0": 2, "cam1": 0}


def test_one_pass_is_one_model_call_and_each_cameras_boxes_carry_the_stamp_and_frame_name_of_their_image():
    calls = []

    def stand_in_model(rgb_images):
        calls.append(len(rgb_images))
        return a_box_as_wide_as_the_images_value(rgb_images)

    sub = SubscriberWithThese([colour("cam1", (7, 11), 110), colour("cam0", (7, 10), 100),
                               colour("cam1", (7, 21), 210)])
    pub, since = RecordingPublisher(), live.SinceLastReport()
    live.detect_on_what_is_waiting(sub, pub, stand_in_model, since, wait_for_the_first_ms=0)

    assert calls == [2], "all cameras with a new image go to the model together"
    assert [topic for topic, _, _ in pub.sent] == ["/detect/cam0/detections", "/detect/cam1/detections"]
    assert [fields["header"] for _, fields, _ in pub.sent] == [
        {"stamp": {"sec": 7, "nanosec": 10}, "frame_id": "cam0_color_optical_frame"},
        {"stamp": {"sec": 7, "nanosec": 21}, "frame_id": "cam1_color_optical_frame"}]
    widths = [detections_of(topic, fields)[2][0][1][2] for topic, fields, _ in pub.sent]
    assert widths == [100.0, 210.0], "each camera's boxes must come from its own newest image"
    assert (dict(since.frames_boxed), dict(since.skipped)) == ({"cam0": 1, "cam1": 1}, {"cam0": 0, "cam1": 1})
    assert since.images_in_a_call == [2] and len(since.model_call_s) == 1


def test_a_pass_with_nothing_waiting_calls_no_model_and_sends_nothing():
    pub, since = RecordingPublisher(), live.SinceLastReport()
    live.detect_on_what_is_waiting(SubscriberWithThese([]), pub, a_box_as_wide_as_the_images_value, since, 0)
    assert pub.sent == [] and since.model_call_s == []


def test_a_frame_with_nothing_found_still_gets_its_boxes_message():
    pub = RecordingPublisher()
    live.detect_on_what_is_waiting(SubscriberWithThese([colour("cam0", (0, 1), 0)]), pub,
                                   lambda rgb_images: [[] for _ in rgb_images], live.SinceLastReport(), 0)
    ((topic, fields, _),) = pub.sent
    assert detections_of(topic, fields) == ("cam0", (0, 1), [])


def test_with_no_colour_image_in_the_whole_period_the_line_says_it_is_waiting_where_and_for_what():
    line = live.report_line(live.SinceLastReport(), 5.0, "tcp://127.0.0.1:5555")
    assert "waiting" in line and "tcp://127.0.0.1:5555" in line and "sim.run" in line


def test_the_report_gives_each_cameras_frames_a_second_and_skipped_and_the_medians_of_the_model_calls():
    since = live.SinceLastReport()
    since.frames_boxed.update({"cam1": 28, "cam0": 25})
    since.skipped.update({"cam0": 3, "cam1": 0})
    since.model_call_s, since.images_in_a_call = [0.050, 0.083, 0.900], [1, 3, 3]
    assert live.report_line(since, 5.0, "tcp://127.0.0.1:5555") == (
        "detect: cam0 5.0 fps (3 skipped), cam1 5.6 fps (0 skipped); model call 83 ms for 3 images (medians)")


def test_a_camera_that_stops_sending_stays_in_the_report_at_zero():
    since = live.SinceLastReport()
    since.frames_boxed.update({"cam0": 5, "cam1": 5})
    since = since.emptied_for_the_same_cameras()
    since.frames_boxed["cam0"] += 5
    since.model_call_s, since.images_in_a_call = [0.025], [1]
    assert "cam1 0.0 fps (0 skipped)" in live.report_line(since, 5.0, "tcp://127.0.0.1:5555")


def test_frames_sent_on_the_feed_come_back_as_boxes_on_the_detectors_own_port(live_cfg, run_dir):
    feed = Publisher("127.0.0.1", "*", QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    _, pub, detect, sub = live.started({**live_cfg, "feed_address": feed.address}, run_dir.name,
                                       load_detector=lambda run_dir, device: a_box_as_wide_as_the_images_value)
    boxes = Subscriber([pub.address], [DETECTIONS_TOPIC_PREFIX], QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    received = None
    for frame in range(1, 101):
        send_frame(feed, "cam0", (0, frame), K, colour_filled_with(frame), DEPTH)
        live.detect_on_what_is_waiting(sub, pub, detect, live.SinceLastReport(), wait_for_the_first_ms=20)
        received = boxes.recv(timeout_ms=20)
        if received is not None:
            break
    for socket in (feed, pub, sub, boxes):
        socket.close()

    assert received is not None, "no boxes message came back"
    camera, (_, frame_of_the_boxes), found = detections_of(*received[:2])
    assert camera == "cam0" and received[1]["header"]["frame_id"] == "cam0_color_optical_frame"
    assert found == [("cube", pytest.approx((0.0, 0.0, float(frame_of_the_boxes), 10.0)), 0.5)], \
        "the boxes must be those of the frame whose stamp they carry"
