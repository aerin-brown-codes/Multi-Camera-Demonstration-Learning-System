import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detect.config import SET_WATCHED_WHILE_TRAINING, load_config
from detect.score import CONFIG_OF_A_RUN, CUT_OFF_OF_A_RUN, PICKED, WEIGHTS_OF_A_RUN, seen_on, sure_enough
from detect.scoring import WRONG, Detection, Judged, judged
from detect.truth import ImageWithTruth, group_between_edges

PICTURES_OF_A_RUN = "mistakes"
FOUND_BGR, MISSED_BGR, FALSE_BGR, DOES_NOT_COUNT_BGR = (0, 200, 0), (0, 0, 255), (255, 0, 255), (160, 160, 160)
THIN, THICK, TEXT_SCALE = 1, 2, 0.45


def kinds_of_mistake(image: ImageWithTruth, detections: list[Detection], outcome: Judged, cfg: dict) -> set[str]:
    kinds = set()
    for truth, was_found in zip(image.truths, outcome.truth_was_found):
        if truth.counts and not was_found:
            how_hidden = "" if truth.visible_fraction is None else \
                f", {group_between_edges(truth.visible_fraction, cfg['visible_fraction'])} visible"
            kinds.add(f"missed {truth.class_name}{how_hidden}")
    kinds.update(f"false {detection.class_name}" for detection, was in zip(detections, outcome.detection_was) if was == WRONG)
    return kinds


def box_drawn(picture: np.ndarray, box, colour, thickness: int, text: str = "") -> None:
    left, top, right, bottom = (round(edge) for edge in box)
    cv2.rectangle(picture, (left, top), (right, bottom), colour, thickness)
    if text:
        cv2.putText(picture, text, (left, max(top - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, colour, THIN, cv2.LINE_AA)


def picture_of(image: ImageWithTruth, detections: list[Detection], outcome: Judged) -> np.ndarray:
    picture = cv2.imread(str(image.path))
    for truth, was_found in zip(image.truths, outcome.truth_was_found):
        if not truth.counts:
            box_drawn(picture, truth.box, DOES_NOT_COUNT_BGR, THIN)
        elif was_found:
            box_drawn(picture, truth.box, FOUND_BGR, THIN)
        else:
            shown = "" if truth.visible_fraction is None else f" {truth.visible_fraction:.0%} visible"
            box_drawn(picture, truth.box, MISSED_BGR, THICK, f"missed {truth.class_name}{shown}")
    for detection, was in zip(detections, outcome.detection_was):
        if was == WRONG:
            box_drawn(picture, detection.box, FALSE_BGR, THICK, f"false {detection.class_name} {detection.confidence:.2f}")
    return picture


def write_pictures(run_dir: Path, detect=None) -> dict[str, dict[str, tuple[int, int]]]:
    cfg = load_config(run_dir / CONFIG_OF_A_RUN)
    cut_off = json.loads((run_dir / CUT_OFF_OF_A_RUN).read_text())[PICKED]
    if detect is None:
        from detect.detector import detector_from_weights
        detect = detector_from_weights(run_dir / WEIGHTS_OF_A_RUN, cfg)
    shutil.rmtree(run_dir / PICTURES_OF_A_RUN, ignore_errors=True)
    drawn_of_all = {}
    for source, (images, detections_per_image) in seen_on(SET_WATCHED_WHILE_TRAINING, cfg, detect).items():
        with_mistake = defaultdict(list)
        for image, detections in zip(images, sure_enough(detections_per_image, cut_off), strict=True):
            outcome = judged(list(image.truths), detections, cfg["score"]["min_iou"])
            for kind in kinds_of_mistake(image, detections, outcome, cfg["score"]):
                with_mistake[kind].append((image, detections, outcome))
        drawn_of_all[source] = {}
        for kind in sorted(with_mistake):
            most = min(cfg["score"]["pictures_of_each_mistake"], len(with_mistake[kind]))
            folder = run_dir / PICTURES_OF_A_RUN / source / kind
            folder.mkdir(parents=True)
            for image, detections, outcome in random.Random(cfg["seed"]).sample(with_mistake[kind], most):
                cv2.imwrite(str(folder / image.path.name), picture_of(image, detections, outcome))
            drawn_of_all[source][kind] = (most, len(with_mistake[kind]))
    return drawn_of_all


def main():
    parser = argparse.ArgumentParser(description="Draw what a run got wrong on the check sets, at its picked cut-off: "
                                                 "missed boxes red, false detections pink, found boxes green.")
    parser.add_argument("--run", type=Path, required=True, help="folder of a run that detect.score has scored")
    run_dir = parser.parse_args().run
    for source, kinds in write_pictures(run_dir).items():
        for kind, (drawn, images) in kinds.items():
            print(f"{source}: {kind}: {drawn} of {images} images drawn")
    print(f"written to {run_dir / PICTURES_OF_A_RUN}/<source>/<kind of mistake>/")


if __name__ == "__main__":
    main()
