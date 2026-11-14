from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from detect import detector
from detect.scoring import Detection


class ModelThatSeesACubeInEveryImage:
    names = {0: "hand", 1: "cube"}

    def __init__(self, weights):
        self.weights = weights
        self.sizes_of_calls = []

    def predict(self, images, **settings):
        self.sizes_of_calls.append(len(images))
        self.images = images
        self.settings = settings
        boxes = SimpleNamespace(cls=torch.tensor([1.0]), xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]), conf=torch.tensor([0.5]))
        return [SimpleNamespace(boxes=boxes) for _ in images]


@pytest.fixture
def cfg_and_model(cfg, monkeypatch):
    made = []
    monkeypatch.setattr(detector, "YOLO", lambda weights: made.append(ModelThatSeesACubeInEveryImage(weights)) or made[-1])
    return {**cfg, "image_size": 864, "device": "cpu"}, made


def test_images_reach_the_model_a_batch_at_a_time_and_every_image_gets_its_detections(cfg_and_model):
    cfg, made = cfg_and_model
    detect = detector.detector_from_weights(Path("best.pt"), cfg)
    detections = list(detect([Path(f"{n}.png") for n in range(10)]))
    assert made[0].sizes_of_calls == [4, 4, 2], "a list given whole would be run as one batch and not fit in memory"
    assert detections == [[Detection("cube", (1.0, 2.0, 3.0, 4.0), 0.5)]] * 10
    assert made[0].settings["imgsz"] == 864
    assert made[0].settings["conf"] == 0.25, "the model must keep everything the lowest cut-off of the list would"


def test_weights_trained_for_other_classes_are_refused(cfg_and_model):
    cfg, _ = cfg_and_model
    with pytest.raises(ValueError, match="trained for"):
        detector.detector_from_weights(Path("best.pt"), {**cfg, "class_names": ["cube", "hand"]})


def test_auto_in_the_config_reaches_the_model_as_a_device_of_this_machine(cfg_and_model, monkeypatch):
    cfg, made = cfg_and_model
    monkeypatch.setattr(detector, "device_to_use", lambda asked: f"picked for {asked}")
    list(detector.detector_from_weights(Path("best.pt"), {**cfg, "device": "auto"})([Path("0.png")]))
    assert made[0].settings["device"] == "picked for auto"


def red_image():
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 0] = 200
    return rgb


def test_arrays_of_the_feed_reach_the_model_with_red_and_blue_swapped_because_the_tool_reads_arrays_as_bgr(
        cfg_and_model, run_dir):
    _, made = cfg_and_model
    rgb = red_image()
    detector.detector_of_a_run(run_dir, "cpu")([rgb])
    (as_the_model_got_it,) = made[0].images
    assert as_the_model_got_it[0, 0].tolist() == [0, 0, 200], "without the swap the red cube reaches the model blue"
    assert rgb[0, 0].tolist() == [200, 0, 0], "the caller's image is left alone"


def test_the_runs_own_folder_supplies_the_weights_the_image_size_the_class_names_and_the_cut_off(
        cfg_and_model, run_dir):
    _, made = cfg_and_model
    detections = detector.detector_of_a_run(run_dir, "cpu")([red_image()])
    assert made[0].weights == run_dir / "weights" / "best.pt"
    assert (made[0].settings["imgsz"], made[0].settings["conf"]) == (640, 0.4)
    assert detections == [[Detection("cube", (1.0, 2.0, 3.0, 4.0), 0.5)]], "class 1 of the run's own list is cube"


def test_arrays_given_together_are_one_call_of_the_model_and_none_is_no_call(cfg_and_model, run_dir):
    _, made = cfg_and_model
    detect = detector.detector_of_a_run(run_dir, "cpu")
    assert len(detect([red_image()] * 4)) == 4
    assert detect([]) == []
    assert made[0].sizes_of_calls == [4]


def test_a_run_whose_weights_were_trained_for_other_classes_is_refused(cfg_and_model, run_dir):
    settings = yaml.safe_load((run_dir / "train.yaml").read_text())
    (run_dir / "train.yaml").write_text(yaml.safe_dump({**settings, "class_names": ["cube", "hand"]}))
    with pytest.raises(ValueError, match="trained for"):
        detector.detector_of_a_run(run_dir, "cpu")


def test_the_device_asked_for_live_is_used_and_not_the_one_the_run_trained_on(cfg_and_model, run_dir, monkeypatch):
    _, made = cfg_and_model
    monkeypatch.setattr(detector, "device_to_use", lambda asked: f"picked for {asked}")
    detector.detector_of_a_run(run_dir, "auto")([red_image()])
    assert made[0].settings["device"] == "picked for auto"
