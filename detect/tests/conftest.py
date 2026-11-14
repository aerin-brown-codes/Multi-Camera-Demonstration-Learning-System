import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from detect.labels import label_line

IMAGE_SIZE = (848, 480)


def write_image_and_labels(root: Path, set_name: str, name: str, classes_and_boxes, suffix=".png") -> Path:
    for kind in ("images", "labels"):
        (root / kind / set_name).mkdir(parents=True, exist_ok=True)
    image = root / "images" / set_name / f"{name}{suffix}"
    cv2.imwrite(str(image), np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8))
    (root / "labels" / set_name / f"{name}.txt").write_text(
        "".join(label_line(class_id, box, *IMAGE_SIZE) + "\n" for class_id, box in classes_and_boxes))
    return image


def write_scene(root: Path, set_name: str, scene_number: int, kind: str, cameras: dict) -> None:
    (root / "scenes" / set_name).mkdir(parents=True, exist_ok=True)
    scene = {"kind": kind, "cameras": {}, "boxes": {}}
    for camera_name, (distance_m, height_deg, boxes) in cameras.items():
        scene["cameras"][camera_name] = {"distance_m": distance_m, "height_deg": height_deg}
        scene["boxes"][camera_name] = boxes
        write_image_and_labels(root, set_name, f"{scene_number:06d}_{camera_name}",
                               [({"hand": 0, "cube": 1}[b["class"]], b["box"]) for b in boxes.values() if b["counts"]])
    (root / "scenes" / set_name / f"{scene_number:06d}.json").write_text(json.dumps(scene))


def box_of_scene(class_name, box, visible_fraction=1.0, counts=True) -> dict:
    return {"class": class_name, "box": box, "visible_fraction": visible_fraction, "counts": counts}


@pytest.fixture
def cfg(tmp_path) -> dict:
    return {
        "runs_dir": str(tmp_path / "runs"), "class_names": ["hand", "cube"], "seed": 0,
        "sources": {"sim": {"root": str(tmp_path / "sim"), "train_images": 4, "truth_from": "scenes"},
                    "real": {"root": str(tmp_path / "real"), "train_images": 4, "truth_from": "labels"}},
        "score": {"cut_offs": [0.25, 0.5, 0.75], "pictures_of_each_mistake": 2, "min_iou": 0.5, "batch": 4, "pass_bar": {"found": 0.9, "right": 0.9},
                  "visible_fraction": [0.1, 0.4, 0.7, 1.0], "camera_distance_m": [0.4, 0.7, 1.0, 1.5],
                  "camera_height_deg": [10, 30, 50, 70, 90]}}


@pytest.fixture
def run_dir(cfg) -> Path:
    run_dir = Path(cfg["runs_dir"]) / "first"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").touch()
    (run_dir / "train.yaml").write_text(yaml.safe_dump({**cfg, "image_size": 640, "device": "never used live"}))
    (run_dir / "cut_off.json").write_text(json.dumps({"picked": 0.4}))
    return run_dir
