import json
import subprocess
import sys

import numpy as np
import pytest
from feed.messages import stamp_fields

from sim.check_lift import (LIFTED, MISSED, NO_DEPTH, Record, bands_of, in_band, progress_line, report,
                            require_poses_and_targets, row)

MIN_OVERLAP = 0.5
BANDS = [0.1, 0.4, 0.7, 1.0]
CAMERA_POSITION = [0.0, 0.0, 2.0]
CUBE_POINT, HAND_POINT = [0.0, 0.0, 1.0], [0.3, 0.0, 1.0]
CUBE_BOX, HAND_BOX = [100, 100, 140, 140], [200, 100, 300, 200]
POSES = ["hands_apart", "cube_lifted", "both_fists"]
EXACT_MM = 1e-6


def camera_at(position):
    T_table_cam = np.eye(4)
    T_table_cam[:3, 3] = position
    return T_table_cam.tolist()


def truth(frame, pose=0, cameras=("cam0",), cube_visible=1.0, hand_counts=True):
    boxes = {"cube": {"class": "cube", "box": CUBE_BOX, "visible_fraction": cube_visible, "counts": True},
             "hand_right": {"class": "hand", "box": HAND_BOX, "visible_fraction": 0.9, "counts": hand_counts}}
    made = {"stamp": stamp_fields((0, frame)), "frame": frame,
            "cameras": {camera: camera_at(CAMERA_POSITION) for camera in cameras},
            "boxes": {camera: boxes for camera in cameras},
            "target_points": {"cube": CUBE_POINT, "hand_right": HAND_POINT},
            "pose": {"index": pose, "name": POSES[pose], "count": len(POSES)}}
    return json.loads(json.dumps(made))


CUBE_AND_HAND_BOXES = [("hand", tuple(map(float, HAND_BOX)), 0.8), ("cube", tuple(map(float, CUBE_BOX)), 0.9)]
CUBE_30_MM_TOWARDS_THE_CAMERA = (1, "cube", (0.0, 0.0, 1.03), 0.9)
HAND_ON_ITS_POINT = (0, "hand", tuple(HAND_POINT), 0.8)


def record_of(frames_kept=60):
    return Record(MIN_OVERLAP, frames_kept)


def add_view(record, frame, camera="cam0", boxes=CUBE_AND_HAND_BOXES,
             positions=(HAND_ON_ITS_POINT, CUBE_30_MM_TOWARDS_THE_CAMERA), **of_the_truth):
    record.add_truth(truth(frame, **of_the_truth))
    record.add_boxes(camera, (0, frame), list(boxes))
    record.add_positions(camera, (0, frame), list(positions))


def by_target(record):
    return {sample.target: sample for sample in record.samples()}


def test_a_lifted_target_gets_its_error_from_truths_camera_and_point():
    record = record_of()
    add_view(record, frame=1)
    cube, hand = by_target(record)["cube"], by_target(record)["hand_right"]
    assert (cube.pose, cube.camera, cube.class_name, cube.outcome) == ("hands_apart", "cam0", "cube", LIFTED)
    assert (cube.error.sideways_mm, cube.error.along_mm) == pytest.approx((0.0, -30.0), abs=EXACT_MM)
    assert hand.outcome == LIFTED and hand.error.distance_mm == pytest.approx(0.0, abs=EXACT_MM)


def test_a_box_whose_id_is_not_in_the_positions_message_is_counted_as_no_depth():
    record = record_of()
    add_view(record, frame=1, positions=[HAND_ON_ITS_POINT])
    assert (by_target(record)["cube"].outcome, by_target(record)["cube"].error) == (NO_DEPTH, None)
    assert by_target(record)["hand_right"].outcome == LIFTED


def test_a_target_with_no_box_is_missed_and_one_that_does_not_count_is_left_out():
    record = record_of()
    add_view(record, frame=1, boxes=[], positions=[], hand_counts=False)
    assert {target: sample.outcome for target, sample in by_target(record).items()} == {"cube": MISSED}


def test_many_frames_of_one_pose_give_one_sample_and_the_later_frame_replaces_the_earlier():
    record = record_of()
    add_view(record, frame=1, positions=[HAND_ON_ITS_POINT])
    add_view(record, frame=2)
    add_view(record, frame=3)
    assert len(record.samples()) == 2, "one per target, however many frames the pose was held"
    assert by_target(record)["cube"].outcome == LIFTED, "the newest frame of the pose is the one that stands"


def test_positions_that_arrive_after_later_truth_and_boxes_still_give_a_sample():
    record = record_of()
    record.add_truth(truth(1))
    record.add_boxes("cam0", (0, 1), CUBE_AND_HAND_BOXES)
    record.add_truth(truth(2))
    record.add_boxes("cam0", (0, 2), [])
    record.add_positions("cam0", (0, 1), [HAND_ON_ITS_POINT, CUBE_30_MM_TOWARDS_THE_CAMERA])
    assert by_target(record)["cube"].outcome == LIFTED


def test_positions_of_a_stamp_that_is_no_longer_kept_or_has_no_boxes_give_no_sample():
    record = record_of(frames_kept=2)
    record.add_truth(truth(1))
    record.add_boxes("cam0", (0, 1), CUBE_AND_HAND_BOXES)
    record.add_truth(truth(2))
    record.add_truth(truth(3))
    record.add_positions("cam0", (0, 1), [CUBE_30_MM_TOWARDS_THE_CAMERA])
    record.add_positions("cam0", (0, 3), [])
    record.add_boxes("cam0", (0, 9), CUBE_AND_HAND_BOXES)
    assert record.samples() == [] and record.views == {}


def test_it_is_done_when_every_pose_was_seen_on_every_camera_also_where_nothing_counted():
    record = record_of()
    assert not record.has_every_pose_on_every_camera(), "nothing seen yet is not done"
    for pose in range(len(POSES)):
        for camera in ("cam0", "cam1"):
            assert not record.has_every_pose_on_every_camera()
            record.add_truth(truth(pose, pose, cameras=("cam0", "cam1"), cube_visible=0.0, hand_counts=False))
            record.add_boxes(camera, (0, pose), [])
            record.add_positions(camera, (0, pose), [])
    assert record.has_every_pose_on_every_camera()


def test_the_all_line_of_a_class_holds_its_three_bands_together_and_each_sample_is_in_one_band():
    record = record_of()
    visible = [0.1, 0.39, 0.4, 0.69, 0.7, 1.0]
    for frame, cube_visible in enumerate(visible):
        add_view(record, frame, pose=frame % len(POSES), camera=f"cam{frame}", cameras=(f"cam{frame}",),
                 cube_visible=cube_visible)
    cubes = [sample for sample in record.samples() if sample.class_name == "cube"]
    bands = bands_of(BANDS)
    assert [label for label, *_ in bands] == ["70 to 100%", "40 to 70%", "10 to 40%"]
    in_each = [[s.visible_fraction for s in cubes if in_band(s, *edges)] for _, *edges in bands]
    assert sorted(map(sorted, in_each)) == [[0.1, 0.39], [0.4, 0.69], [0.7, 1.0]], \
        "a band holds its lower edge, and only the last holds its upper one"
    printed = report(record, BANDS, worst_listed=5).splitlines()
    assert row("cube", "all", cubes) in printed and row("cube", "all", cubes).split()[2] == "6"
    for (label, *edges), fractions in zip(bands, in_each):
        line = row("cube", label, [s for s in cubes if in_band(s, *edges)])
        assert line in printed and line.split()[4] == str(len(fractions)) == "2"


def test_the_row_counts_every_view_and_takes_its_errors_from_the_lifted_ones():
    record = record_of()
    add_view(record, frame=1, pose=0)
    add_view(record, frame=2, pose=1, positions=[HAND_ON_ITS_POINT])
    add_view(record, frame=3, pose=2, boxes=[], positions=[])
    cubes = [sample for sample in record.samples() if sample.class_name == "cube"]
    assert row("cube", "all", cubes).split() == ["cube", "all", "3", "1", "1", "0.0", "0.0", "0.0", "-30",
                                                 "-30", "to", "-30"]
    assert row("cube", "all", cubes[2:]).split() == ["cube", "all", "1", "1", "0", "-", "-", "-", "-", "-"]


def test_the_report_lists_the_worst_by_distance_and_counts_boxes_on_no_target():
    record = record_of()
    a_box_on_nothing = ("cube", (400.0, 300.0, 440.0, 340.0), 0.3)
    add_view(record, frame=1, boxes=[*CUBE_AND_HAND_BOXES, a_box_on_nothing])
    printed = report(record, BANDS, worst_listed=1)
    assert printed.splitlines()[0] == "lift on the sim feed: 3 poses, 1 cameras"
    worst = printed.split("worst 1 by distance:\n")[1].splitlines()[0]
    assert worst.split() == ["hands_apart", "cam0", "cube", "100%", "visible", "sideways", "0.0", "along", "-30",
                             "distance", "30.0", "mm"]
    assert "boxes on no target that counts: 1" in printed


def test_stopped_early_it_names_the_poses_not_seen_and_the_cameras_that_sent_no_positions():
    record = record_of()
    record.add_truth(truth(1, pose=1, cameras=("cam0", "cam1")))
    record.add_boxes("cam0", (0, 1), CUBE_AND_HAND_BOXES)
    record.add_boxes("cam1", (0, 1), CUBE_AND_HAND_BOXES)
    record.add_positions("cam0", (0, 1), [CUBE_30_MM_TOWARDS_THE_CAMERA])
    printed = report(record, BANDS, worst_listed=5)
    assert "stopped early: 3 of 3 poses were not seen on every camera (poses number 0, 1, 2, counted from 0)" in printed
    assert "no positions came for cam1: detect.live has no camera pose for them" in printed


def test_a_finished_report_says_nothing_about_stopping_early():
    record = record_of()
    for pose in range(len(POSES)):
        add_view(record, frame=pose, pose=pose)
    printed = report(record, BANDS, worst_listed=5)
    assert record.has_every_pose_on_every_camera()
    assert "stopped early" not in printed and "no positions came" not in printed


def test_truth_without_poses_or_targets_is_refused_by_what_it_lacks():
    require_poses_and_targets(truth(1))
    without = {key: value for key, value in truth(1).items() if key not in ("pose", "target_points")}
    with pytest.raises(SystemExit, match="no pose and no target_points.*sim.run --scene cube_hands"):
        require_poses_and_targets(without)


def test_the_progress_line_says_what_it_is_waiting_for():
    record = record_of()
    assert progress_line(record, "tcp://sim", "tcp://detect") == \
        "check_lift: waiting for truth from tcp://sim (sim.run should be running)"
    record.add_truth(truth(1))
    assert progress_line(record, "tcp://sim", "tcp://detect") == \
        "check_lift: waiting for positions from tcp://detect (detect.live should be running)"
    record.add_boxes("cam0", (0, 1), [])
    record.add_positions("cam0", (0, 1), [])
    assert progress_line(record, "tcp://sim", "tcp://detect") == "check_lift: 1 of 3 poses seen on every camera"


def test_the_report_and_the_viewer_load_nothing_of_the_detector():
    code = ("import sys, sim.check_lift, sim.view\n"
            "print(sorted({'detect', 'torch', 'ultralytics'} & set(sys.modules)))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
