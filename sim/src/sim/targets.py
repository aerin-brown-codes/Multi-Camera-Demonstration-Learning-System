from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.boxes import BoxLimits, target_box_or_none_when_out_of_view
from sim.render import NO_GEOM, RigRenderer

NO_TARGET = -1


@dataclass(frozen=True)
class Target:
    name: str
    class_name: str
    body: str


def load_targets(path) -> list[Target]:
    cfg = yaml.safe_load(Path(path).read_text())
    return [Target(name, t["class"], t["body"]) for name, t in cfg.get("targets", {}).items()]


def body_and_every_body_below_it(model: mujoco.MjModel, top: int) -> list[int]:
    bodies = {top}
    for body_numbered_after_its_parent in range(top + 1, model.nbody):
        if model.body_parentid[body_numbered_after_its_parent] in bodies:
            bodies.add(body_numbered_after_its_parent)
    return sorted(bodies)


def target_of_each_geom(model: mujoco.MjModel, targets: list[Target]) -> np.ndarray:
    target_of_body = np.full(model.nbody, NO_TARGET)
    for index, target in enumerate(targets):
        for body in body_and_every_body_below_it(model, model.body(target.body).id):
            if target_of_body[body] != NO_TARGET:
                raise ValueError(f"body '{model.body(body).name}' is in two targets: "
                                 f"'{targets[target_of_body[body]].name}' and '{target.name}'")
            target_of_body[body] = index
    return target_of_body[model.geom_bodyid]


class TargetBoxes:
    def __init__(self, model: mujoco.MjModel, renderer: RigRenderer, targets: list[Target], limits: BoxLimits):
        self.targets = targets
        self._renderer = renderer
        self._limits = limits
        self._target_of_geom = target_of_each_geom(model, targets)
        self._geoms_of = [self._target_of_geom == index for index in range(len(targets))]

    def in_every_camera(self, data: mujoco.MjData) -> dict[str, dict[str, dict]]:
        boxes = {}
        for cam in self._renderer.rig.cameras:
            geom = self._renderer.geom_at_each_pixel(data, cam)
            target_at_each_pixel = np.where(geom == NO_GEOM, NO_TARGET, self._target_of_geom[geom])
            boxes[cam.name] = {}
            for index, target in enumerate(self.targets):
                alone = self._renderer.geom_at_each_pixel(data, cam, drawing_only=self._geoms_of[index]) != NO_GEOM
                box = target_box_or_none_when_out_of_view(target_at_each_pixel == index, alone, self._limits)
                if box is not None:
                    boxes[cam.name][target.name] = {"class": target.class_name, **asdict(box)}
        return boxes
