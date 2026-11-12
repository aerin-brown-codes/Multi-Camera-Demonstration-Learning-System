import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from sim.boxes import BoxLimits
from sim.draws import chosen_by_share
from sim.pose_variation import PoseVariation, VariedPose
from sim.poses import Poser, load_scene_poses
from sim.render import RigRenderer
from sim.rig import Rig, load_profiles, replace_camera_of_the_same_name
from sim.scene_variation import (PlacedCamera, SceneLook, apply_look, look_in_words,
                                 look_with_extras_put_somewhere_else, random_camera, random_look)
from sim.targets import NO_TARGET, TargetBoxes, load_targets, target_of_each_geom
from sim.world import build_model, top_of_box_shaped_table

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = Path(__file__).parents[3] / "datasets" / "sim"
KINDS_KEEPING_AN_IMAGE_WITH_NO_TARGET_IN_IT = {"sparse"}
FPS_OF_STILL_IMAGES = 0.0


@dataclass(frozen=True)
class Scene:
    description: dict
    images: dict[str, np.ndarray]


@dataclass(frozen=True)
class BuiltScene:
    varied: VariedPose
    look: SceneLook
    cameras: list[PlacedCamera]
    rig: Rig
    model: mujoco.MjModel
    data: mujoco.MjData
    builds: int


def label_line(class_id: int, x0_y0_x1_y1, width: int, height: int) -> str:
    x0, y0, x1, y1 = x0_y0_x1_y1
    return (f"{class_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")


def label_lines(boxes_of_one_camera: dict[str, dict], class_ids: dict[str, int], width: int, height: int) -> list[str]:
    return [label_line(class_ids[b["class"]], b["box"], width, height)
            for b in boxes_of_one_camera.values() if b["counts"]]


def scenes_in_set(images: int, cameras_per_scene: int) -> int:
    return math.ceil(images / cameras_per_scene)


class SceneMaker:
    def __init__(self, config_path: Path):
        config_dir = Path(config_path).parent
        self.cfg = cfg = yaml.safe_load(Path(config_path).read_text())
        sim_cfg = yaml.safe_load((config_dir / cfg["sim_config"]).read_text())
        scene_config = config_dir / "scenes" / f"{cfg['scene']}.yaml"
        self.profile = load_profiles(config_dir / sim_cfg["rig"])[cfg["camera_profile"]]
        self._allowed_overlap_m = sim_cfg["allowed_overlap_m"]
        self._box_limits = BoxLimits(**sim_cfg["truth_boxes"])
        self._scene_poses = load_scene_poses(scene_config)
        self._targets = load_targets(scene_config)

        self._model_without_looks = build_model(cfg["scene"], Rig(FPS_OF_STILL_IMAGES, []))
        self._data_without_looks = mujoco.MjData(self._model_without_looks)
        self._table = top_of_box_shaped_table(self._model_without_looks, cfg["table_geom"])
        self._geoms_of_targets = np.flatnonzero(
            target_of_each_geom(self._model_without_looks, self._targets) != NO_TARGET)
        self._pose_variation = PoseVariation(
            Poser(self._model_without_looks, self._scene_poses), self._data_without_looks, self._scene_poses,
            self._table, cfg["poses"], self._allowed_overlap_m)

    def kind_of_scene(self, set_name: str, number: int) -> str:
        return chosen_by_share(self._rng_of_scene(set_name, number), self.cfg["scene_kinds"])

    def scene(self, set_name: str, number: int) -> Scene:
        cfg = self.cfg
        rng = self._rng_of_scene(set_name, number)
        kind = chosen_by_share(rng, cfg["scene_kinds"])
        built = self.build_with_nothing_overlapping(rng, kind)
        renderer = RigRenderer(built.model, built.rig)
        try:
            target_boxes = TargetBoxes(built.model, renderer, self._targets, self._box_limits)
            images, boxes, camera_words = {}, {}, {}
            for placed in built.cameras:
                name = placed.camera.name
                for camera_draws in range(1, cfg["camera_tries"] + 1):
                    boxes[name] = target_boxes.in_camera(built.data, placed.camera)
                    if kind in KINDS_KEEPING_AN_IMAGE_WITH_NO_TARGET_IN_IT or any(
                            b["counts"] for b in boxes[name].values()):
                        break
                    if camera_draws < cfg["camera_tries"]:
                        placed = random_camera(rng, name, self.profile, self._table, cfg["cameras"])
                        replace_camera_of_the_same_name(built.model, built.data, built.rig, placed.camera)
                images[name] = renderer.render_camera(built.data, placed.camera)[0]
                camera_words[name] = {
                    "position": placed.camera.position.tolist(), "aim": placed.aim.tolist(),
                    "distance_m": float(placed.distance_m), "height_deg": float(placed.height_deg),
                    "T_world_cam": placed.camera.T_world_cam.tolist(), "draws": camera_draws}
        finally:
            renderer.close()
        return Scene({
            "set": set_name, "scene": number, "kind": kind, "made_from": built.varied.made_from,
            "mirrored": built.varied.mirrored, "scene_builds": built.builds,
            "image_size": [self.profile.width, self.profile.height],
            "cameras": camera_words, "boxes": boxes, "look": look_in_words(built.look)}, images)

    def build_with_nothing_overlapping(self, rng: np.random.Generator, kind: str) -> BuiltScene:
        cfg = self.cfg
        varied = self._pose_variation.draw(rng, kind)
        places_of_targets_xy = self._places_of_targets_over_the_table_xy()
        look = random_look(rng, self._table, places_of_targets_xy, cfg["look"])
        cameras = [random_camera(rng, f"cam{i}", self.profile, self._table, cfg["cameras"])
                   for i in range(cfg["cameras_per_scene"])]
        rig = Rig(FPS_OF_STILL_IMAGES, [placed.camera for placed in cameras])
        for builds in range(1, cfg["extras_tries"] + 1):
            model = build_model(cfg["scene"], rig, partial(
                apply_look, look=look, table_geom=cfg["table_geom"], floor_geom=cfg["floor_geom"],
                skin_cfg=cfg["look"]["skin"], backdrop_px=cfg["look"]["backdrop_px"]))
            data = mujoco.MjData(model)
            poser = Poser(model, self._scene_poses)
            poser.place(data, varied.pose)
            overlaps = poser.overlaps(data, self._allowed_overlap_m)
            if not overlaps:
                return BuiltScene(varied, look, cameras, rig, model, data, builds)
            sunk_into_something = {owner for overlap in overlaps for owner in overlap.owners}
            look = look_with_extras_put_somewhere_else(
                rng, look, sunk_into_something, builds < cfg["look"]["extras"]["place_tries"], self._table,
                places_of_targets_xy, cfg["look"]["extras"])
        raise RuntimeError(f"a {kind} scene" + (f" made from '{varied.made_from}'" if varied.made_from else "")
                           + f" still had {'; '.join(map(str, overlaps))} after its extra objects were put somewhere "
                           f"else {cfg['extras_tries']} times")

    def _rng_of_scene(self, set_name: str, number: int) -> np.random.Generator:
        return np.random.default_rng([self.cfg["sets"][set_name]["seed"], number])

    def _places_of_targets_over_the_table_xy(self) -> np.ndarray:
        places = self._data_without_looks.geom_xpos[self._geoms_of_targets, :2]
        return places[[self._table.is_under(xy) for xy in places]]


def scene_file(out: Path, set_name: str, number: int) -> Path:
    return out / "scenes" / set_name / f"{number:06d}.json"


def write_scene(out: Path, scene: Scene, class_ids: dict[str, int]) -> None:
    d = scene.description
    set_name, number, (width, height) = d["set"], d["scene"], d["image_size"]
    for folder in ("images", "labels", "scenes"):
        (out / folder / set_name).mkdir(parents=True, exist_ok=True)
    for cam, rgb in scene.images.items():
        stem = f"{number:06d}_{cam}"
        cv2.imwrite(str(out / "images" / set_name / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lines = label_lines(d["boxes"][cam], class_ids, width, height)
        (out / "labels" / set_name / f"{stem}.txt").write_text("".join(line + "\n" for line in lines))
    scene_file(out, set_name, number).write_text(json.dumps(d))


def write_set(maker: SceneMaker, out: Path, set_name: str, only_the_first: int | None = None) -> int:
    cfg = maker.cfg
    total = scenes_in_set(cfg["sets"][set_name]["images"], cfg["cameras_per_scene"])
    numbers = range(total if only_the_first is None else min(total, only_the_first))
    to_write = [n for n in numbers if not scene_file(out, set_name, n).exists()]
    print(f"dataset: {set_name}: writing {len(to_write)} scenes, keeping {len(numbers) - len(to_write)} "
          f"already in {out}", flush=True)
    started = report = time.perf_counter()
    for done, number in enumerate(to_write, start=1):
        write_scene(out, maker.scene(set_name, number), cfg["class_ids"])
        if time.perf_counter() - report >= cfg["progress_every_s"] or done == len(to_write):
            report = time.perf_counter()
            print(f"dataset: {set_name}: {done} of {len(to_write)} scenes, {(report - started) / done:.2f} s per scene",
                  flush=True)
    return len(to_write)


def what_is_in_a_set(out: Path, set_name: str) -> str:
    kinds, labels, images, images_without_a_label = Counter(), Counter(), 0, 0
    for path in sorted((out / "scenes" / set_name).glob("*.json")):
        d = json.loads(path.read_text())
        kinds[d["kind"]] += 1
        for boxes in d["boxes"].values():
            counted = [b["class"] for b in boxes.values() if b["counts"]]
            labels.update(counted)
            images += 1
            images_without_a_label += not counted
    scenes = sum(kinds.values())
    return (f"{set_name}: {scenes} scenes ("
            + ", ".join(f"{kind} {count / max(scenes, 1):.0%}" for kind, count in sorted(kinds.items()))
            + f"), {images} images, {images_without_a_label} with no label, labels: "
            + ", ".join(f"{count} {name}" for name, count in sorted(labels.items())))


def main():
    parser = argparse.ArgumentParser(
        description="Write the labelled sim images. A scene that is already written is kept, so a stopped run "
                    "carries on where it was; delete a set's folders to make it again.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--set", action="append", dest="sets", help="write only this set (can be repeated)")
    parser.add_argument("--scenes", type=int, help="write only the first N scenes of each set")
    args = parser.parse_args()

    maker = SceneMaker(args.config)
    for set_name in args.sets or maker.cfg["sets"]:
        write_set(maker, args.out, set_name, args.scenes)
        print("dataset: " + what_is_in_a_set(args.out, set_name))


if __name__ == "__main__":
    main()
