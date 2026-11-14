from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detect.config import load_config
from detect.device import device_to_use
from detect.score import CONFIG_OF_A_RUN, WEIGHTS_OF_A_RUN, picked_cut_off
from detect.scoring import Detection

Detector = Callable[[list[Path]], Iterator[list[Detection]]]
DetectorOfArrays = Callable[[list[np.ndarray]], list[list[Detection]]]


def model_trained_for(weights: Path, class_names: list[str]) -> YOLO:
    model = YOLO(weights)
    names_in_weights = [model.names[class_id] for class_id in sorted(model.names)]
    if names_in_weights != class_names:
        raise ValueError(f"{weights} was trained for {names_in_weights}, the config says {class_names}")
    return model


def detections_in(result, class_names: list[str]) -> list[Detection]:
    boxes = result.boxes
    return [Detection(class_names[int(class_id)], tuple(box), float(confidence))
            for class_id, box, confidence in zip(boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist())]


def detector_from_weights(weights: Path, cfg: dict) -> Detector:
    model = model_trained_for(weights, cfg["class_names"])
    device = device_to_use(cfg["device"])
    least_sure_any_cut_off_keeps = min(cfg["score"]["cut_offs"])

    def detections_per_image(images: list[Path]) -> Iterator[list[Detection]]:
        a_list_given_whole_is_run_as_one_batch = cfg["score"]["batch"]
        for start in range(0, len(images), a_list_given_whole_is_run_as_one_batch):
            together = [str(image) for image in images[start:start + a_list_given_whole_is_run_as_one_batch]]
            for result in model.predict(together, conf=least_sure_any_cut_off_keeps, imgsz=cfg["image_size"],
                                        device=device, verbose=False):
                yield detections_in(result, cfg["class_names"])

    return detections_per_image


def detector_of_a_run(run_dir: Path, device: str | int) -> DetectorOfArrays:
    cfg = load_config(run_dir / CONFIG_OF_A_RUN)
    cut_off = picked_cut_off(run_dir)
    model = model_trained_for(run_dir / WEIGHTS_OF_A_RUN, cfg["class_names"])
    device = device_to_use(device)

    def detections_per_image(rgb_images: list[np.ndarray]) -> list[list[Detection]]:
        if not rgb_images:
            return []
        as_the_tool_reads_arrays = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgb_images]
        results = model.predict(as_the_tool_reads_arrays, conf=cut_off, imgsz=cfg["image_size"], device=device,
                                verbose=False)
        return [detections_in(result, cfg["class_names"]) for result in results]

    return detections_per_image
