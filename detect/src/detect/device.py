import torch

PICKED_ON_THE_MACHINE = "auto"


def device_to_use(asked: str | int) -> str | int:
    if asked != PICKED_ON_THE_MACHINE:
        return asked
    if torch.cuda.is_available():
        return 0
    return "mps" if torch.backends.mps.is_available() else "cpu"
