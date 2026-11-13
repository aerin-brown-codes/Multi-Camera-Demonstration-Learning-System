import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/repo")
RUNS = REPO / "runs"
CODE = REPO / "detect" / "src"
CONFIGS = CODE / "detect" / "configs"
SCORES_OF_A_FINISHED_RUN = "scores.md"
WEIGHTS_TO_CARRY_ON_FROM = Path("weights") / "last.pt"
START, CARRY_ON, SKIP = "start", "carry on", "skip"
SECONDS_BETWEEN_LOOKS_AT_THE_LOG = 30
LOOKS_BETWEEN_LINES_ABOUT_THE_MACHINE = 2
ASKED_FOR_ONE_QUEUE_ONLY = ("RUNS", "COMMIT", "RESUME", "KEEP_THE_MACHINE")


def asked_and_forgotten_so_the_next_queue_of_this_session_starts_clean() -> dict[str, str]:
    return {setting: os.environ.pop(setting, "") for setting in ASKED_FOR_ONE_QUEUE_ONLY}


def runs_asked_for(text: str) -> list[tuple[str, str]]:
    return [tuple(part.strip() for part in run.split("=")) for run in text.split(",")]


def queue_that_can_start(asked: dict[str, str]) -> list[tuple[str, str]]:
    if not asked["RUNS"] or not asked["COMMIT"]:
        raise KeyError("pass --env RUNS=<name>=<config file>,... and --env COMMIT=<commit id of the code tar>")
    queue = runs_asked_for(asked["RUNS"])
    missing = [config for _, config in queue if not (CONFIGS / config).exists()]
    if missing:
        raise FileNotFoundError(f"not in {CONFIGS}: {missing}")
    return queue


def what_to_do(run_dir: Path, carrying_on: bool) -> str:
    if not carrying_on:
        return START
    if (run_dir / SCORES_OF_A_FINISHED_RUN).exists():
        return SKIP
    return CARRY_ON if (run_dir / WEIGHTS_TO_CARRY_ON_FROM).exists() else START


def last_state_of_each_line(text: str) -> list[str]:
    return [line.split("\r")[-1] for line in text.split("\n") if line.strip()]


def line_about_the_machine() -> str:
    gpu = "none"
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    return f"machine: gpu {gpu}; processors busy {os.getloadavg()[0]:.1f} of {os.cpu_count()}"


def train_and_score(name: str, config: str, action: str, commit: str, carrying_on: bool) -> int:
    run_dir, log_file = RUNS / name, RUNS / f"{name}.log"
    if action == START and run_dir.exists():
        if not carrying_on:
            raise FileExistsError(f"{run_dir} already holds a run; pick another name, or pass RESUME=1 to carry it on")
        print(f"run: {name}: was cut before its first epoch ended, nothing to carry on from; starting it again", flush=True)
        shutil.rmtree(run_dir)
    command = [sys.executable, "-m", "detect.train", "--config", str(CONFIGS / config), "--name", name]
    if action == CARRY_ON:
        command.append("--resume")
    with open(log_file, "ab") as log:
        child = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(CODE), "COMMIT": commit},
                                 stdout=log, stderr=subprocess.STDOUT)
        looks = 0
        with open(log_file, "rb") as written_so_far:
            written_so_far.seek(0, os.SEEK_END)
            while True:
                finished = child.poll() is not None
                for line in last_state_of_each_line(written_so_far.read().decode(errors="replace")):
                    print(line, flush=True)
                if finished:
                    return child.returncode
                if looks % LOOKS_BETWEEN_LINES_ABOUT_THE_MACHINE == 0:
                    log.write((line_about_the_machine() + "\n").encode())
                    log.flush()
                looks += 1
                time.sleep(SECONDS_BETWEEN_LOOKS_AT_THE_LOG)


def train_the_queue(queue: list[tuple[str, str]], commit: str, carrying_on: bool) -> dict[str, str]:
    how_each_ended = {}
    for name, config in queue:
        action = what_to_do(RUNS / name, carrying_on)
        print(f"run: {name} ({config}): {action}", flush=True)
        if action == SKIP:
            how_each_ended[name] = "was already finished"
            continue
        try:
            exit_code = train_and_score(name, config, action, commit, carrying_on)
            how_each_ended[name] = "finished" if exit_code == 0 else f"FAILED with exit code {exit_code}, see {name}.log"
        except Exception as error:
            how_each_ended[name] = f"FAILED: {error!r}"
    return how_each_ended


def let_drive_finish_writing_then_give_the_machine_back() -> None:
    from google.colab import drive, runtime
    try:
        drive.flush_and_unmount()
    finally:
        runtime.unassign()


if __name__ == "__main__":
    asked = asked_and_forgotten_so_the_next_queue_of_this_session_starts_clean()
    queue = queue_that_can_start(asked)
    try:
        for name, ended in train_the_queue(queue, asked["COMMIT"], bool(asked["RESUME"])).items():
            print(f"run: {name}: {ended}", flush=True)
    finally:
        if not asked["KEEP_THE_MACHINE"]:
            let_drive_finish_writing_then_give_the_machine_back()
