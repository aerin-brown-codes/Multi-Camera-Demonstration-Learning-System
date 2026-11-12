from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from sim.rig import Rig, add_cameras

SCENES = Path(__file__).parent / "scenes"


@dataclass(frozen=True)
class TableTop:
    centre_xy: np.ndarray
    half_size_xy: np.ndarray
    z: float

    def random_point_xy(self, rng: np.random.Generator, margin_m: float = 0.0) -> np.ndarray:
        return self.centre_xy + rng.uniform(-1, 1, 2) * (self.half_size_xy - margin_m)

    def is_under(self, xy, margin_m: float = 0.0) -> bool:
        return bool(np.all(np.abs(np.asarray(xy) - self.centre_xy) <= self.half_size_xy - margin_m))


def build_model(scene: str, rig: Rig, change: Callable[[mujoco.MjSpec], None] | None = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENES / f"{scene}.xml"))
    if change is not None:
        change(spec)
    add_cameras(spec, rig)
    return spec.compile()


def top_of_box_shaped_table(model: mujoco.MjModel, geom: str) -> TableTop:
    table = model.geom(geom)
    return TableTop(table.pos[:2].copy(), table.size[:2].copy(), float(table.pos[2] + table.size[2]))
