import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from detect.labels import class_and_box_px_of_line, images_of, label_file_of
from detect.scoring import Truth

SCENE_KIND, CAMERA_DISTANCE, CAMERA_HEIGHT = "scene kind", "camera distance (m)", "camera height (deg)"


@dataclass(frozen=True)
class ImageWithTruth:
    path: Path
    truths: tuple[Truth, ...]
    groups: dict[str, str]


def group_between_edges(value: float, edges: list[float]) -> str:
    if value < edges[0]:
        return f"below {edges[0]:g}"
    for low, high in zip(edges, edges[1:]):
        if value < high or (high == edges[-1] and value == high):
            return f"{low:g} to {high:g}"
    return f"above {edges[-1]:g}"


def class_ids_in_label_file(image: Path) -> list[int]:
    return sorted(int(line.split()[0]) for line in label_file_of(image).read_text().splitlines())


def images_with_truth_from_scenes(root: Path, set_name: str, class_names: list[str], cfg: dict) -> list[ImageWithTruth]:
    images = []
    for scene_file in sorted((root / "scenes" / set_name).glob("*.json")):
        scene = json.loads(scene_file.read_text())
        for camera_name, camera in scene["cameras"].items():
            image = root / "images" / set_name / f"{scene_file.stem}_{camera_name}.png"
            if not image.exists():
                raise FileNotFoundError(f"{scene_file} describes {image.name}, which is not in {image.parent}")
            truths = tuple(Truth(box["class"], tuple(box["box"]), box["counts"], box["visible_fraction"])
                           for box in scene["boxes"][camera_name].values() if box["box"] is not None)
            ids_that_count = sorted(class_names.index(truth.class_name) for truth in truths if truth.counts)
            if ids_that_count != class_ids_in_label_file(image):
                raise ValueError(f"the label file of {image} does not hold the boxes that count in {scene_file}; "
                                 f"class_names {class_names} or the set itself is out of step with step 3")
            images.append(ImageWithTruth(image, truths, {
                SCENE_KIND: scene["kind"],
                CAMERA_DISTANCE: group_between_edges(camera["distance_m"], cfg["camera_distance_m"]),
                CAMERA_HEIGHT: group_between_edges(camera["height_deg"], cfg["camera_height_deg"])}))
    return images


def images_with_truth_from_labels(root: Path, set_name: str, class_names: list[str]) -> list[ImageWithTruth]:
    images = []
    for image in images_of(root, set_name):
        height, width = cv2.imread(str(image)).shape[:2]
        lines = label_file_of(image).read_text().splitlines()
        truths = tuple(Truth(class_names[class_id], box)
                       for class_id, box in (class_and_box_px_of_line(line, width, height) for line in lines))
        images.append(ImageWithTruth(image, truths, {}))
    return images


WAYS_TO_READ_TRUTH = {
    "scenes": images_with_truth_from_scenes,
    "labels": lambda root, set_name, class_names, cfg: images_with_truth_from_labels(root, set_name, class_names)}
