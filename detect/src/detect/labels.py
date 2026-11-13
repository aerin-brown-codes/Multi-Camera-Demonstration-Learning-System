from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg"}


def label_line(class_id: int, box_px, width: int, height: int) -> str:
    x0, y0, x1, y1 = box_px
    return (f"{class_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")


def class_and_box_px_of_line(line: str, width: int, height: int) -> tuple[int, tuple[float, float, float, float]]:
    class_id, xc, yc, w, h = line.split()
    xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
    return int(class_id), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def images_of(root: Path, set_name: str) -> list[Path]:
    return sorted(p for p in (root / "images" / set_name).iterdir() if p.suffix in IMAGE_SUFFIXES)


def label_file_of(image: Path) -> Path:
    root, set_name = image.parents[2], image.parent.name
    return root / "labels" / set_name / f"{image.stem}.txt"
