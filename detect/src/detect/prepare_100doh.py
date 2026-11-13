import argparse
import shutil
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

import cv2
import numpy as np

from detect.config import CONFIGS, from_repo, load_config
from detect.labels import label_line

DEFAULT_CONFIG = CONFIGS / "100doh.yaml"
SHORTEST_SIDE_OF_A_BOX_PX = 1
FOLDER_OF_A_SET_BEING_WRITTEN = "{}.partial"


def kept_label_lines(annotation_xml: str, keep: dict[str, int]) -> tuple[list[str], tuple[int, int]]:
    annotation = ElementTree.fromstring(annotation_xml)
    width, height = int(annotation.findtext("size/width")), int(annotation.findtext("size/height"))
    lines = []
    for labelled in annotation.iter("object"):
        if labelled.findtext("name") not in keep:
            continue
        x0, y0, x1, y1 = (float(labelled.findtext(f"bndbox/{corner}")) for corner in ("xmin", "ymin", "xmax", "ymax"))
        x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, width), min(y1, height)
        if min(x1 - x0, y1 - y0) >= SHORTEST_SIDE_OF_A_BOX_PX:
            lines.append(label_line(keep[labelled.findtext("name")], (x0, y0, x1, y1), width, height))
    return lines, (width, height)


def size_of_image(image_bytes: bytes) -> tuple[int, int] | None:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else (image.shape[1], image.shape[0])


def write_set(download: zipfile.ZipFile, cfg: dict, set_name: str, index_of_set: int, report=print) -> None:
    out, inside, wanted = from_repo(cfg["out"]), cfg["inside_zip"], cfg["sets"][set_name]
    if (out / "images" / set_name).exists():
        report(f"{set_name}: already in {out}; delete images/{set_name} and labels/{set_name} to prepare it again")
        return
    being_written = {kind: out / kind / FOLDER_OF_A_SET_BEING_WRITTEN.format(set_name) for kind in ("labels", "images")}
    for folder in being_written.values():
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)

    names = sorted(download.read(f"{inside}/ImageSets/Main/{wanted['from']}.txt").decode().split())
    rng = np.random.default_rng([cfg["seed"], index_of_set])
    written, passed_over = 0, 0
    for name in (names[i] for i in rng.permutation(len(names))):
        if written == wanted["images"]:
            break
        image_bytes = download.read(f"{inside}/JPEGImages/{name}.jpg")
        lines, size_in_annotation = kept_label_lines(download.read(f"{inside}/Annotations/{name}.xml").decode(), cfg["keep"])
        if size_of_image(image_bytes) != size_in_annotation:
            passed_over += 1
            continue
        (being_written["labels"] / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines))
        (being_written["images"] / f"{name}.jpg").write_bytes(image_bytes)
        written += 1
    if written < wanted["images"]:
        raise RuntimeError(f"{set_name}: only {written} usable images in the download's '{wanted['from']}' list, "
                           f"{wanted['images']} asked for")
    for kind, folder in being_written.items():
        folder.rename(out / kind / set_name)
    report(f"{set_name}: {written} images written, {passed_over} passed over because the image is not the size "
           f"its annotation says")


def main():
    parser = argparse.ArgumentParser(description="Take the hand images and labels we use out of the 100DOH download.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args().config)
    with zipfile.ZipFile(from_repo(cfg["zip"])) as download:
        for index_of_set, set_name in enumerate(cfg["sets"]):
            write_set(download, cfg, set_name, index_of_set)


if __name__ == "__main__":
    main()
