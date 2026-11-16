import sys

import numpy as np
import pytest
from feed.detections import DETECTIONS_TOPIC_PREFIX, detections_of, detections_topic, positions_of
from feed.frames import send_frame
from feed.messages import camera_info_message, color_message, color_optical_frame, depth_message
from feed.pipe import Publisher, Subscriber
from feed.transforms import TABLE_FRAME, static_transforms_message

from detect import detector, live
from detect.scoring import Detection

WIDTH, HEIGHT = 64, 48
K = np.eye(3)
QUEUE_LARGE_ENOUGH_TO_NEVER_DROP = 200


def colour_filled_with(value):
    return np.full((HEIGHT, WIDTH, 3), value, np.uint8)


def depth_filled_with(depth_mm):
    return np.full((HEIGHT, WIDTH), depth_mm, np.uint16)


WALL_MM = 1000
LIFT = live.lift_as_configured({"lift": {"middle_fraction": 0.5, "min_valid_fraction": 0.5}})
CAMERA_AT_THE_TABLES_ORIGIN = np.eye(4)


def camera_moved_to(x, y, z):
    T_table_cam = np.eye(4)
    T_table_cam[:3, 3] = x, y, z
    return T_table_cam


def as_received(topic, fields, data):
    return topic, fields, None if data is None else data.tobytes()


def colour(camera, stamp, value):
    return as_received(*color_message(camera, stamp, colour_filled_with(value)))


def depth(camera, stamp, depth_mm=WALL_MM):
    image = depth_mm if isinstance(depth_mm, np.ndarray) else depth_filled_with(depth_mm)
    return as_received(*depth_message(camera, stamp, image))


def frame(camera, stamp, value, depth_mm=WALL_MM):
    return [as_received(*camera_info_message(camera, stamp, K, WIDTH, HEIGHT)), colour(camera, stamp, value),
            depth(camera, stamp, depth_mm)]


def poses(**T_table_cam_of_camera):
    return as_received(*static_transforms_message((0, 0), TABLE_FRAME, {
        color_optical_frame(camera): T_table_cam for camera, T_table_cam in T_table_cam_of_camera.items()}))


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


def one_pass(messages, detect=None, seen=None, since=None):
    pub = RecordingPublisher()
    seen, since = seen or live.SeenOnTheFeed(), since or live.SinceLastReport()
    live.detect_on_what_is_waiting(SubscriberWithThese(messages), pub, detect or a_box_as_wide_as_the_images_value,
                                   LIFT, seen, since, wait_for_the_first_ms=0)
    return pub.sent, seen, since


@pytest.fixture
def live_cfg(cfg, run_dir):
    return {"runs_dir": cfg["runs_dir"], "run": run_dir.name, "device": "cpu",
            "feed_address": "tcp://127.0.0.1:5555", "publish": {"host": "127.0.0.1", "port": "*"},
            "queue_messages": QUEUE_LARGE_ENOUGH_TO_NEVER_DROP, "report_every_s": 5,
            "lift": {"middle_fraction": 0.5, "min_valid_fraction": 0.5}}


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


def test_of_several_waiting_frames_the_newest_complete_one_of_each_camera_is_used_and_the_others_are_counted():
    first, second, third = (0, 1), (0, 2), (0, 3)
    newest, skipped = live.newest_complete_frame_of_each_camera([
        *frame("cam0", first, 1), *frame("cam1", first, 1), *frame("cam0", second, 2), *frame("cam0", third, 3),
        colour("cam1", second, 2), depth("cam1", third),
    ], live.SeenOnTheFeed())
    assert {camera: (found.stamp, int(found.rgb[0, 0, 0])) for camera, found in newest.items()} == {
        "cam0": (third, 3), "cam1": (first, 1)}, "a colour image whose depth image never came is not a frame"
    assert dict(skipped) == {"cam0": 2, "cam1": 0}


def test_a_frame_whose_colour_and_depth_arrive_in_two_reads_of_the_socket_is_still_put_together():
    lens_numbers, colour_image, depth_image = frame("cam0", (0, 1), 30)
    sent, seen, since = one_pass([lens_numbers, colour_image])
    assert sent == [] and since.model_call_s == []
    sent, _, _ = one_pass([depth_image], seen=seen, since=since)
    assert [detections_of(topic, fields) for topic, fields, _ in sent] == [
        ("cam0", (0, 1), [("cube", pytest.approx((0.0, 0.0, 30.0, 10.0)), 0.5)])]


def test_one_pass_is_one_model_call_and_each_cameras_boxes_carry_the_stamp_and_frame_name_of_their_image():
    calls = []

    def stand_in_model(rgb_images):
        calls.append(len(rgb_images))
        return a_box_as_wide_as_the_images_value(rgb_images)

    sent, _, since = one_pass([*frame("cam1", (7, 11), 110), *frame("cam0", (7, 10), 100),
                               *frame("cam1", (7, 21), 210)], stand_in_model)

    assert calls == [2], "all cameras with a new frame go to the model together"
    assert [topic for topic, _, _ in sent] == ["/detect/cam0/detections", "/detect/cam1/detections"]
    assert [fields["header"] for _, fields, _ in sent] == [
        {"stamp": {"sec": 7, "nanosec": 10}, "frame_id": color_optical_frame("cam0")},
        {"stamp": {"sec": 7, "nanosec": 21}, "frame_id": color_optical_frame("cam1")}]
    widths = [detections_of(topic, fields)[2][0][1][2] for topic, fields, _ in sent]
    assert widths == [100.0, 210.0], "each camera's boxes must come from its own newest image"
    assert (dict(since.frames_boxed), dict(since.skipped)) == ({"cam0": 1, "cam1": 1}, {"cam0": 0, "cam1": 1})
    assert since.images_in_a_call == [2] and len(since.model_call_s) == 1


def test_a_pass_with_nothing_waiting_calls_no_model_and_sends_nothing():
    sent, _, since = one_pass([])
    assert sent == [] and since.model_call_s == []


def test_a_frame_with_nothing_found_still_gets_its_boxes_message_and_an_empty_positions_message():
    sent, _, _ = one_pass([poses(cam0=CAMERA_AT_THE_TABLES_ORIGIN), *frame("cam0", (0, 1), 0)],
                          lambda rgb_images: [[] for _ in rgb_images])
    (boxes_topic, boxes, _), (positions_topic, positions, _) = sent
    assert detections_of(boxes_topic, boxes) == ("cam0", (0, 1), [])
    assert positions_of(positions_topic, positions) == ("cam0", (0, 1), [])


def test_colour_images_whose_depth_never_comes_get_no_boxes_and_the_camera_is_named():
    sent, _, since = one_pass([colour("cam1", (0, 1), 10), colour("cam1", (0, 2), 20), *frame("cam0", (0, 2), 20)])
    assert [topic for topic, _, _ in sent] == ["/detect/cam0/detections"]
    assert "detect: no depth images for cam1, so no boxes" in live.lines_about_what_is_missing(since)


def test_each_cameras_positions_follow_its_boxes_and_are_lifted_with_the_depth_image_of_the_same_stamp():
    sent, _, since = one_pass([
        poses(cam0=CAMERA_AT_THE_TABLES_ORIGIN, cam1=camera_moved_to(1.0, 2.0, 3.0)),
        *frame("cam0", (0, 1), 20, depth_mm=1000), *frame("cam0", (0, 2), 20, depth_mm=2000),
        colour("cam0", (0, 3), 20), *frame("cam1", (0, 2), 20, depth_mm=500)])

    assert [topic for topic, _, _ in sent] == ["/detect/cam0/detections", "/detect/cam0/detections_3d",
                                               "/detect/cam1/detections", "/detect/cam1/detections_3d"]
    assert [fields["header"] for _, fields, _ in sent[1::2]] == [
        {"stamp": {"sec": 0, "nanosec": 2}, "frame_id": "table"}] * 2, "positions are in table coordinates"
    (_, _, (lifted_by_cam0,)), (_, _, (lifted_by_cam1,)) = [positions_of(topic, fields) for topic, fields, _ in sent[1::2]]
    centre_of_the_box_as_the_lens_counts_it = np.array([10.0 - 0.5, 5.0 - 0.5, 1.0])
    assert lifted_by_cam0 == ("0", "cube", pytest.approx(tuple(2.0 * centre_of_the_box_as_the_lens_counts_it)), 0.5), \
        "the depth must be that of stamp 2, the frame the box was found in"
    assert lifted_by_cam1 == ("0", "cube", pytest.approx(tuple(0.5 * centre_of_the_box_as_the_lens_counts_it
                                                               + [1.0, 2.0, 3.0])), 0.5)
    assert (since.boxes, since.boxes_lifted) == (2, 2) and live.lines_about_what_is_missing(since) == []


def test_a_camera_whose_pose_has_not_come_gets_its_boxes_and_no_positions_message():
    sent, _, since = one_pass([poses(cam1=CAMERA_AT_THE_TABLES_ORIGIN), *frame("cam0", (0, 1), 20),
                               *frame("cam1", (0, 1), 20), *frame("cam2", (0, 1), 20)])
    assert [topic for topic, _, _ in sent] == ["/detect/cam0/detections", "/detect/cam1/detections",
                                               "/detect/cam1/detections_3d", "/detect/cam2/detections"]
    assert (since.boxes, since.boxes_lifted) == (3, 1)
    assert live.lines_about_what_is_missing(since) == [
        "detect: no camera pose for cam0, cam2 on /tf_static, so boxes only"]


def test_a_box_over_pixels_without_a_depth_reading_is_absent_and_the_other_boxes_keep_their_ids():
    with_a_hole = depth_filled_with(WALL_MM)
    with_a_hole[:, 20:30] = 0

    def three_boxes_side_by_side(rgb_images):
        return [[Detection("cube", (0.0, 0.0, 10.0, 10.0), 0.9), Detection("hand", (20.0, 0.0, 30.0, 10.0), 0.8),
                 Detection("hand", (40.0, 0.0, 50.0, 10.0), 0.7)] for _ in rgb_images]

    sent, _, since = one_pass([poses(cam0=CAMERA_AT_THE_TABLES_ORIGIN), *frame("cam0", (0, 1), 0, with_a_hole)],
                              three_boxes_side_by_side)
    (boxes_topic, boxes, _), (positions_topic, positions, _) = sent
    assert [detection["id"] for detection in boxes["detections"]] == ["0", "1", "2"]
    assert [(box_id, class_name, confidence) for box_id, class_name, _, confidence
            in positions_of(positions_topic, positions)[2]] == [("0", "cube", 0.9), ("2", "hand", 0.7)]
    assert (since.boxes, since.boxes_lifted) == (3, 2)


def test_a_pose_that_changes_is_used_from_the_next_frame_on():
    seen = live.SeenOnTheFeed()
    heights = []
    for stamp, z in (((0, 1), 0.0), ((0, 2), 5.0)):
        sent, _, _ = one_pass([poses(cam0=camera_moved_to(0.0, 0.0, z)), *frame("cam0", stamp, 20)], seen=seen)
        heights.append(positions_of(*sent[1][:2])[2][0][2][2])
    sent, _, _ = one_pass(frame("cam0", (0, 3), 20), seen=seen)
    heights.append(positions_of(*sent[1][:2])[2][0][2][2])
    assert heights == pytest.approx([1.0, 6.0, 6.0]), "the newest pose received is the one used, until another comes"


def test_with_no_frame_in_the_whole_period_the_line_says_it_is_waiting_where_and_for_what():
    line = live.report_line(live.SinceLastReport(), 5.0, "tcp://127.0.0.1:5555")
    assert "waiting for frames" in line and "tcp://127.0.0.1:5555" in line and "sim.run" in line


def test_the_report_gives_each_cameras_frames_a_second_and_skipped_the_medians_of_the_model_calls_and_boxes_lifted():
    since = live.SinceLastReport()
    since.frames_boxed.update({"cam1": 28, "cam0": 25})
    since.skipped.update({"cam0": 3, "cam1": 0})
    since.model_call_s, since.images_in_a_call = [0.050, 0.083, 0.900], [1, 3, 3]
    since.boxes, since.boxes_lifted = 3000, 2996
    assert live.report_line(since, 5.0, "tcp://127.0.0.1:5555") == (
        "detect: cam0 5.0 fps (3 skipped), cam1 5.6 fps (0 skipped); model call 83 ms for 3 images (medians); "
        "2996 of 3000 boxes lifted")


def test_a_camera_that_stops_sending_stays_in_the_report_at_zero():
    since = live.SinceLastReport()
    since.frames_boxed.update({"cam0": 5, "cam1": 5})
    since.boxes, since.cameras_without_a_pose = 10, {"cam0", "cam1"}
    since = since.emptied_for_the_same_cameras()
    since.frames_boxed["cam0"] += 5
    since.model_call_s, since.images_in_a_call = [0.025], [1]
    assert "cam1 0.0 fps (0 skipped)" in live.report_line(since, 5.0, "tcp://127.0.0.1:5555")
    assert since.boxes == 0 and live.lines_about_what_is_missing(since) == [], "a new period starts from nothing"


def test_frames_and_a_pose_sent_on_the_feed_come_back_as_boxes_and_positions_on_the_detectors_own_port(
        live_cfg, run_dir):
    feed = Publisher("127.0.0.1", "*", QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    _, pub, detect, sub = live.started({**live_cfg, "feed_address": feed.address}, run_dir.name,
                                       load_detector=lambda run_dir, device: a_box_as_wide_as_the_images_value)
    from_the_detector = Subscriber([pub.address], [DETECTIONS_TOPIC_PREFIX], QUEUE_LARGE_ENOUGH_TO_NEVER_DROP)
    seen, received = live.SeenOnTheFeed(), []
    for number in range(1, 101):
        feed.send(*static_transforms_message((0, number), TABLE_FRAME,
                                             {color_optical_frame("cam0"): camera_moved_to(0.0, 0.0, 10.0)}))
        send_frame(feed, "cam0", (0, number), K, colour_filled_with(number), depth_filled_with(100 * number))
        live.detect_on_what_is_waiting(sub, pub, detect, live.lift_as_configured(live_cfg), seen,
                                       live.SinceLastReport(), wait_for_the_first_ms=20)
        while len(received) < 2 and (message := from_the_detector.recv(timeout_ms=20)) is not None:
            if received or message[0] == detections_topic("cam0"):
                received.append(message)
        if len(received) == 2:
            break
    for socket in (feed, pub, sub, from_the_detector):
        socket.close()

    assert len(received) == 2, "no boxes message followed by a positions message came back"
    (boxes_topic, boxes, _), (positions_topic, positions, _) = received
    camera, (_, frame_of_the_boxes), found = detections_of(boxes_topic, boxes)
    assert camera == "cam0" and boxes["header"]["frame_id"] == color_optical_frame("cam0")
    assert found == [("cube", pytest.approx((0.0, 0.0, float(frame_of_the_boxes), 10.0)), 0.5)], \
        "the boxes must be those of the frame whose stamp they carry"
    _, stamp_of_the_positions, ((box_id, class_name, point, confidence),) = positions_of(positions_topic, positions)
    assert stamp_of_the_positions == (0, frame_of_the_boxes) and (box_id, class_name, confidence) == ("0", "cube", 0.5)
    assert point[2] == pytest.approx(10.0 + 0.1 * frame_of_the_boxes), \
        "the position must be lifted with the depth image and the pose sent with that frame"
