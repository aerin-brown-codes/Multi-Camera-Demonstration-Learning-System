from pathlib import Path

import numpy as np
import yaml

from detect.config import SET_TRAINED_ON, SET_WATCHED_WHILE_TRAINING, from_repo
from detect.labels import images_of, label_file_of


def picked_at_random(rng: np.random.Generator, images: list[Path], count: int) -> list[Path]:
    if count > len(images):
        raise ValueError(f"{count} images asked for but only {len(images)} are in {images[0].parent if images else 'the set'}")
    return [images[i] for i in sorted(rng.choice(len(images), count, replace=False))]


def training_images(cfg: dict) -> list[Path]:
    images = []
    for index, source in enumerate(cfg["sources"].values()):
        rng_of_this_source_alone = np.random.default_rng([cfg["seed"], index])
        images += picked_at_random(rng_of_this_source_alone, images_of(from_repo(source["root"]), SET_TRAINED_ON),
                                   source["train_images"])
    return images


def check_images(cfg: dict) -> list[Path]:
    return [image for source in cfg["sources"].values()
            for image in images_of(from_repo(source["root"]), SET_WATCHED_WHILE_TRAINING)]


def write_lists_and_data_file(run_dir: Path, cfg: dict) -> Path:
    lists = {"train": training_images(cfg), "val": check_images(cfg)}
    data = {"names": dict(enumerate(cfg["class_names"]))}
    for name_the_tool_knows, images in lists.items():
        without_label_file = [image for image in images if not label_file_of(image).exists()]
        if without_label_file:
            raise FileNotFoundError(f"{len(without_label_file)} images have no label file, the first is "
                                    f"{without_label_file[0]}; the tool would train on them as if they showed nothing")
        list_file = run_dir / f"{name_the_tool_knows}.txt"
        list_file.write_text("".join(f"{image}\n" for image in images))
        data[name_the_tool_knows] = str(list_file)
    data_file = run_dir / "data.yaml"
    data_file.write_text(yaml.safe_dump(data, sort_keys=False))
    return data_file
