from collections.abc import Callable, Iterator
from pathlib import Path

from ultralytics import YOLO

from detect.device import device_to_use
from detect.scoring import Detection

Detector = Callable[[list[Path]], Iterator[list[Detection]]]


def detector_from_weights(weights: Path, cfg: dict) -> Detector:
    model = YOLO(weights)
    names_in_weights = [model.names[class_id] for class_id in sorted(model.names)]
    if names_in_weights != cfg["class_names"]:
        raise ValueError(f"{weights} was trained for {names_in_weights}, the config says {cfg['class_names']}")

    device = device_to_use(cfg["device"])

    def detections_per_image(images: list[Path]) -> Iterator[list[Detection]]:
        a_list_given_whole_is_run_as_one_batch = cfg["score"]["batch"]
        for start in range(0, len(images), a_list_given_whole_is_run_as_one_batch):
            together = [str(image) for image in images[start:start + a_list_given_whole_is_run_as_one_batch]]
            for result in model.predict(together, conf=cfg["score"]["confidence"], imgsz=cfg["image_size"],
                                        device=device, verbose=False):
                boxes = result.boxes
                yield [Detection(cfg["class_names"][int(class_id)], tuple(box), float(confidence))
                       for class_id, box, confidence in zip(boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist())]

    return detections_per_image
