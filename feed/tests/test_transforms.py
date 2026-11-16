import json

import numpy as np
import pytest

from feed.messages import color_message, color_optical_frame
from feed.transforms import (TABLE_FRAME, TF_STATIC_TOPIC, quaternion_xyzw_of, rotation_of, static_transforms_message,
                             static_transforms_of)

STAMP = (0, 233333333)
NO_ROTATION = np.eye(3)
HALF_TURN_ABOUT = {"x": np.diag([1.0, -1.0, -1.0]), "y": np.diag([-1.0, 1.0, -1.0]), "z": np.diag([-1.0, -1.0, 1.0])}


def looking_at_the_table_from(position) -> np.ndarray:
    forward = -np.asarray(position, float) / np.linalg.norm(position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


AS_THE_RIGS_CAMERAS_LOOK = {f"cam{i}": looking_at_the_table_from(position) for i, position in enumerate(
    [(0.495, 0.495, 0.5), (-0.495, 0.495, 0.5), (-0.495, -0.495, 0.5), (0.495, -0.495, 0.5)])}


def pose(R, position=(0.495, -0.3, 1.25)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, position
    return T


def over_the_wire(topic, fields, data):
    assert data is None, "a pose message has no pixels"
    return topic, json.loads(json.dumps(fields))


@pytest.mark.parametrize("R", [NO_ROTATION, *HALF_TURN_ABOUT.values(), *AS_THE_RIGS_CAMERAS_LOOK.values()],
                         ids=["no rotation", *(f"half turn about {axis}" for axis in HALF_TURN_ABOUT),
                              *AS_THE_RIGS_CAMERAS_LOOK])
def test_a_pose_read_back_is_the_pose_that_was_written(R):
    written = {"cam0_color_optical_frame": pose(R), "cam1_color_optical_frame": pose(R.T, (-1.0, 2.0, 0.5))}
    _, fields = over_the_wire(*static_transforms_message(STAMP, TABLE_FRAME, written))
    read = static_transforms_of(fields, TABLE_FRAME)
    assert read.keys() == written.keys()
    for child_frame, T in written.items():
        assert np.abs(read[child_frame] - T).max() < 1e-12, child_frame


def test_a_half_turn_has_no_w_and_still_gives_a_whole_quaternion():
    for axis, R in HALF_TURN_ABOUT.items():
        assert quaternion_xyzw_of(R).tolist() == [float(axis == part) for part in "xyz"] + [0.0]


def test_the_quaternion_is_written_x_y_z_w():
    quarter_turn_about_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert quaternion_xyzw_of(quarter_turn_about_z) == pytest.approx([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], abs=1e-15)
    assert rotation_of([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]) == pytest.approx(quarter_turn_about_z, abs=1e-15)


def test_the_message_is_ros_tfmessage_field_for_field():
    topic, fields = over_the_wire(*static_transforms_message(
        STAMP, TABLE_FRAME, {"cam0_color_optical_frame": pose(NO_ROTATION, (0.495, 0.495, 1.25))}))
    assert topic == TF_STATIC_TOPIC == "/tf_static"
    assert fields == {"transforms": [{
        "header": {"stamp": {"sec": 0, "nanosec": 233333333}, "frame_id": "table"},
        "child_frame_id": "cam0_color_optical_frame",
        "transform": {"translation": {"x": 0.495, "y": 0.495, "z": 1.25},
                      "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}]}


def test_a_pose_given_in_another_frame_is_refused_by_name():
    _, fields = over_the_wire(*static_transforms_message(STAMP, "board", {"cam2_color_optical_frame": pose(NO_ROTATION)}))
    with pytest.raises(ValueError, match="cam2_color_optical_frame.*'board'.*'table'"):
        static_transforms_of(fields, TABLE_FRAME)


def test_a_cameras_pose_is_found_by_the_frame_name_its_images_carry():
    _, image_fields, _ = color_message("cam3", STAMP, np.zeros((2, 2, 3), np.uint8))
    assert image_fields["header"]["frame_id"] == color_optical_frame("cam3") == "cam3_color_optical_frame"
