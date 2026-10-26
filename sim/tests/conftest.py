from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

import sim
from sim.render import RigRenderer
from sim.rig import load_rig
from sim.world import build_model

RIG_YAML = Path(sim.__file__).parent / "configs" / "rig.yaml"


@pytest.fixture(scope="module")
def tabletop():
    rig = load_rig(RIG_YAML)
    model = build_model("tabletop", rig)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = RigRenderer(model, rig)
    yield SimpleNamespace(rig=rig, model=model, data=data, frames=renderer.render(data))
    renderer.close()
