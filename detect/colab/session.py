import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ON_DRIVE = Path("/content/drive/MyDrive/capstone")
DATA_ON_DRIVE = ON_DRIVE / "data"
RUNS_ON_DRIVE = ON_DRIVE / "runs"
CODE_TAR = Path("/content/code.tar")
REPO = Path("/content/repo")
DATA_TARS = ("sim.tar", "100doh.tar")
PRETRAINED_WEIGHTS = "yolo26s.pt"
TRAINING_TOOL = "ultralytics"


def version_in_the_lock(lock_text: str, package: str) -> str:
    return next(entry["version"] for entry in tomllib.loads(lock_text)["package"] if entry["name"] == package)


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def set_up_the_machine() -> None:
    needed = [CODE_TAR, DATA_ON_DRIVE / PRETRAINED_WEIGHTS, *(DATA_ON_DRIVE / tar for tar in DATA_TARS)]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError(f"not there: {missing}. Is Drive mounted (colab drivemount) and the code tar uploaded?")

    if (REPO / "runs").is_symlink():
        (REPO / "runs").unlink()
    shutil.rmtree(REPO, ignore_errors=True)
    REPO.mkdir()
    run("tar", "-xf", str(CODE_TAR), "-C", str(REPO))
    for tar in DATA_TARS:
        print(f"session: unpacking {tar} from Drive onto the machine's own disk", flush=True)
        run("tar", "-xf", str(DATA_ON_DRIVE / tar), "-C", str(REPO))
    (REPO / "weights").mkdir()
    shutil.copy(DATA_ON_DRIVE / PRETRAINED_WEIGHTS, REPO / "weights" / PRETRAINED_WEIGHTS)

    version = version_in_the_lock((REPO / "uv.lock").read_text(), TRAINING_TOOL)
    run(sys.executable, "-m", "pip", "install", "--quiet", f"{TRAINING_TOOL}=={version}")

    RUNS_ON_DRIVE.mkdir(exist_ok=True)
    (REPO / "runs").symlink_to(RUNS_ON_DRIVE)


def report_what_the_machine_has() -> None:
    for images in sorted((REPO / "datasets").glob("*/images/*")):
        print(f"session: {images.relative_to(REPO)}: {sum(1 for _ in images.iterdir())} images")
    print(f"session: {os.cpu_count()} processors")
    versions = "import torch, ultralytics; print('session: torch', torch.__version__, 'ultralytics', " \
               "ultralytics.__version__, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
    print(subprocess.run([sys.executable, "-c", versions], capture_output=True, text=True, check=True).stdout, end="")


if __name__ == "__main__":
    set_up_the_machine()
    report_what_the_machine_has()
