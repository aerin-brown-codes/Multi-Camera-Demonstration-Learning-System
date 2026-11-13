import zipfile

import cv2
import numpy as np
import pytest

from detect.labels import class_and_box_px_of_line, images_of, label_file_of
from detect.prepare_100doh import kept_label_lines, write_set

INSIDE = "download/VOC2007"
WIDTH, HEIGHT = 640, 360
HAND_PX = (200, 100, 320, 180)


def annotation(objects, size=(WIDTH, HEIGHT)) -> str:
    listed = "".join(
        f"<object><name>{name}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
        f"</bndbox></object>" for name, (x0, y0, x1, y1) in objects)
    return f"<annotation><size><width>{size[0]}</width><height>{size[1]}</height></size>{listed}</annotation>"


def image_with_a_white_hand() -> bytes:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[HAND_PX[1]:HAND_PX[3], HAND_PX[0]:HAND_PX[2]] = 255
    return cv2.imencode(".jpg", image)[1].tobytes()


@pytest.fixture
def download_and_cfg(tmp_path):
    names = {"train": [f"video_frame{n:03d}" for n in range(12)], "test": [f"other_frame{n:03d}" for n in range(6)]}
    zip_file = tmp_path / "download.zip"
    with zipfile.ZipFile(zip_file, "w") as download:
        for list_name, frames in names.items():
            download.writestr(f"{INSIDE}/ImageSets/Main/{list_name}.txt", "\n".join(frames) + "\n")
            for frame in frames:
                download.writestr(f"{INSIDE}/JPEGImages/{frame}.jpg", image_with_a_white_hand())
                download.writestr(f"{INSIDE}/Annotations/{frame}.xml",
                                  annotation([("hand", HAND_PX), ("targetobject", (300, 150, 400, 250))]))
    cfg = {"zip": str(zip_file), "inside_zip": INSIDE, "out": str(tmp_path / "out"), "seed": 0, "keep": {"hand": 0},
           "sets": {"train": {"from": "train", "images": 5}, "test": {"from": "test", "images": 3}}}
    return zipfile.ZipFile(zip_file), cfg, names


def write_all(download, cfg, report=lambda line: None):
    for index, set_name in enumerate(cfg["sets"]):
        write_set(download, cfg, set_name, index, report)


def test_converted_label_drawn_on_its_image_sits_on_the_hand(download_and_cfg, tmp_path):
    download, cfg, _ = download_and_cfg
    write_all(download, cfg)
    for image_file in images_of(tmp_path / "out", "train"):
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        (line,) = label_file_of(image_file).read_text().splitlines()
        class_id, (x0, y0, x1, y1) = class_and_box_px_of_line(line, WIDTH, HEIGHT)
        assert class_id == 0
        inside = np.zeros(image.shape, bool)
        inside[round(y0):round(y1), round(x0):round(x1)] = True
        assert image[inside].min() > 200, "everything inside the box is hand"
        assert image[~inside].max() < 60, "nothing outside the box is hand"


def test_each_set_gets_the_number_asked_for_from_the_downloads_own_list(download_and_cfg, tmp_path):
    download, cfg, names = download_and_cfg
    write_all(download, cfg)
    for set_name, count in (("train", 5), ("test", 3)):
        written = [image.stem for image in images_of(tmp_path / "out", set_name)]
        assert len(written) == count and set(written) <= set(names[set_name])


def test_same_seed_picks_the_same_images_and_another_seed_others(download_and_cfg, tmp_path):
    download, cfg, _ = download_and_cfg
    picks = []
    for out, seed in (("a", 0), ("b", 0), ("c", 1)):
        write_all(download, {**cfg, "out": str(tmp_path / out), "seed": seed})
        picks.append([image.name for image in images_of(tmp_path / out, "train")])
    assert picks[0] == picks[1] != picks[2]


def test_set_already_prepared_is_left_alone(download_and_cfg, tmp_path):
    download, cfg, _ = download_and_cfg
    write_all(download, cfg)
    before = [image.name for image in images_of(tmp_path / "out", "train")]
    said = []
    write_all(download, {**cfg, "seed": 1}, said.append)
    assert [image.name for image in images_of(tmp_path / "out", "train")] == before
    assert "already in" in said[0]


def test_set_stopped_halfway_is_started_again_from_nothing(download_and_cfg, tmp_path):
    download, cfg, _ = download_and_cfg
    left_behind = tmp_path / "out" / "images" / "train.partial"
    left_behind.mkdir(parents=True)
    (left_behind / "stale.jpg").write_bytes(b"")
    write_all(download, cfg)
    assert not left_behind.exists()
    assert "stale.jpg" not in [image.name for image in images_of(tmp_path / "out", "train")]


def test_image_that_is_not_the_size_its_annotation_says_is_passed_over(download_and_cfg, tmp_path):
    download, cfg, names = download_and_cfg
    download.close()
    with zipfile.ZipFile(cfg["zip"], "a") as more:
        more.writestr(f"{INSIDE}/ImageSets/Main/odd.txt", "odd_frame\n" + "\n".join(names["test"][:2]) + "\n")
        more.writestr(f"{INSIDE}/JPEGImages/odd_frame.jpg", image_with_a_white_hand())
        more.writestr(f"{INSIDE}/Annotations/odd_frame.xml", annotation([("hand", HAND_PX)], size=(1280, 720)))
    cfg["sets"] = {"train": {"from": "odd", "images": 2}}
    said = []
    write_all(zipfile.ZipFile(cfg["zip"]), cfg, said.append)
    assert sorted(image.stem for image in images_of(tmp_path / "out", "train")) == sorted(names["test"][:2])
    assert "1 passed over" in said[0]


def test_asking_for_more_images_than_the_list_has_is_refused_and_nothing_is_left_looking_done(download_and_cfg, tmp_path):
    download, cfg, _ = download_and_cfg
    cfg["sets"]["train"]["images"] = 13
    with pytest.raises(RuntimeError, match="only 12 usable images"):
        write_all(download, cfg)
    assert not (tmp_path / "out" / "images" / "train").exists()


def test_only_kept_names_become_lines_and_boxes_are_cut_at_the_image_edge():
    lines, size = kept_label_lines(annotation([("targetobject", (10, 10, 50, 50)), ("hand", (600, 300, 700, 400)),
                                               ("hand", (650, 10, 660, 20))]), {"hand": 0})
    assert size == (WIDTH, HEIGHT)
    assert len(lines) == 1, "the object is dropped, and so is a hand box that lies outside the image"
    assert class_and_box_px_of_line(lines[0], WIDTH, HEIGHT)[1] == pytest.approx((600, 300, 640, 360), abs=1e-3)
