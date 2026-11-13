from pathlib import Path

import pytest

from detect.labels import class_and_box_px_of_line, images_of, label_file_of, label_line
from conftest import write_image_and_labels


def test_label_line_is_the_one_step_3_writes_for_the_same_box():
    assert label_line(0, (0, 83, 76, 166), 848, 480) == "0 0.044811 0.259375 0.089623 0.172917"


def test_line_read_back_gives_the_class_and_the_box_in_pixels():
    class_id, box = class_and_box_px_of_line(label_line(1, (110, 154, 196, 229), 848, 480), 848, 480)
    assert class_id == 1
    assert box == pytest.approx((110, 154, 196, 229), abs=1e-3)


def test_label_file_sits_under_labels_in_the_same_set_with_the_same_name():
    assert label_file_of(Path("/d/sim/images/test/000001_cam2.png")) == Path("/d/sim/labels/test/000001_cam2.txt")


def test_images_of_a_set_are_sorted_and_only_images(tmp_path):
    for name in ("b", "a"):
        write_image_and_labels(tmp_path, "train", name, [])
    (tmp_path / "images" / "train" / "notes.cache").write_text("")
    assert [image.name for image in images_of(tmp_path, "train")] == ["a.png", "b.png"]
