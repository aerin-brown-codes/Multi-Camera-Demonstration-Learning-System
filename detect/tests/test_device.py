import pytest

from detect import device


@pytest.mark.parametrize("gpu, mps, picked", [(True, True, 0), (True, False, 0), (False, True, "mps"), (False, False, "cpu")])
def test_auto_takes_the_gpu_first_then_the_mac_then_the_processor(monkeypatch, gpu, mps, picked):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: gpu)
    monkeypatch.setattr(device.torch.backends.mps, "is_available", lambda: mps)
    assert device.device_to_use("auto") == picked


def test_device_named_in_the_config_is_used_whatever_the_machine_has(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    assert device.device_to_use("cpu") == "cpu"
