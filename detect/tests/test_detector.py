from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from detect import detector
from detect.scoring import Detection


class ModelThatSeesACubeInEveryImage:
    names = {0: "hand", 1: "cube"}

    def __init__(self, weights):
        self.sizes_of_calls = []

    def predict(self, images, **settings):
        self.sizes_of_calls.append(len(images))
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
    assert made[0].settings["conf"] == 0.25 and made[0].settings["imgsz"] == 864


def test_weights_trained_for_other_classes_are_refused(cfg_and_model):
    cfg, _ = cfg_and_model
    with pytest.raises(ValueError, match="trained for"):
        detector.detector_from_weights(Path("best.pt"), {**cfg, "class_names": ["cube", "hand"]})


def test_auto_in_the_config_reaches_the_model_as_a_device_of_this_machine(cfg_and_model, monkeypatch):
    cfg, made = cfg_and_model
    monkeypatch.setattr(detector, "device_to_use", lambda asked: f"picked for {asked}")
    list(detector.detector_from_weights(Path("best.pt"), {**cfg, "device": "auto"})([Path("0.png")]))
    assert made[0].settings["device"] == "picked for auto"
