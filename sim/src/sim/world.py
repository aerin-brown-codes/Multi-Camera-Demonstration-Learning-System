from pathlib import Path

import mujoco

from sim.rig import Rig, add_cameras

SCENES = Path(__file__).parent / "scenes"


def build_model(scene: str, rig: Rig) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENES / f"{scene}.xml"))
    add_cameras(spec, rig)
    return spec.compile()
