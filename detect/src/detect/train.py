import argparse
import os
import shutil
import subprocess
from pathlib import Path

from ultralytics import YOLO

from detect.config import CONFIGS, REPO, from_repo, load_config
from detect.device import device_to_use
from detect.image_lists import write_lists_and_data_file
from detect.score import CONFIG_OF_A_RUN, as_markdown, score_run

DEFAULT_CONFIG = CONFIGS / "train.yaml"
WEIGHTS_TO_CARRY_ON_FROM = Path("weights") / "last.pt"
COMMIT_OF_A_RUN, COMMIT_GIVEN_WHERE_THE_CODE_HAS_NO_GIT = "commit.txt", "COMMIT"


def commit_of_the_code() -> str:
    return os.environ.get(COMMIT_GIVEN_WHERE_THE_CODE_HAS_NO_GIT) or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def start(config_file: Path, run_dir: Path) -> None:
    cfg = load_config(config_file)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already holds a run; pick another --name, or carry it on with --resume")
    commit = commit_of_the_code()
    run_dir.mkdir(parents=True)
    (run_dir / COMMIT_OF_A_RUN).write_text(commit + "\n")
    shutil.copy(config_file, run_dir / CONFIG_OF_A_RUN)
    data_file = write_lists_and_data_file(run_dir, cfg)
    pretrained = from_repo(cfg["model"])
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    YOLO(pretrained).train(data=str(data_file), imgsz=cfg["image_size"], epochs=cfg["epochs"],
                           patience=cfg["patience"], batch=cfg["batch"], workers=cfg["workers"],
                           device=device_to_use(cfg["device"]), seed=cfg["seed"],
                           project=str(run_dir.parent), name=run_dir.name, exist_ok=True)


def carry_on(run_dir: Path) -> None:
    YOLO(run_dir / WEIGHTS_TO_CARRY_ON_FROM).train(resume=True)


def main():
    parser = argparse.ArgumentParser(description="Train on the mix of images the config asks for, then score.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", required=True, help="the run is kept in <runs_dir>/<name>")
    parser.add_argument("--resume", action="store_true", help="carry on a run that was stopped, with its own config")
    args = parser.parse_args()
    run_dir = from_repo(load_config(args.config)["runs_dir"]) / args.name
    if args.resume:
        carry_on(run_dir)
    else:
        start(args.config, run_dir)
    print(as_markdown(score_run(run_dir)))


if __name__ == "__main__":
    main()
