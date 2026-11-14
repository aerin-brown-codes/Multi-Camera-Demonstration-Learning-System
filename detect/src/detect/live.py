import argparse
import signal
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from feed.detections import detections_message
from feed.messages import CAMERA_TOPIC_PREFIX, COLOR, camera_and_kind, image_of, stamp_of
from feed.pipe import Publisher, Subscriber

from detect.config import CONFIGS, from_repo, load_config
from detect.score import CONFIG_OF_A_RUN, CUT_OFF_OF_A_RUN, WEIGHTS_OF_A_RUN, picked_cut_off

DEFAULT_CONFIG = CONFIGS / "live.yaml"
END_OF_A_COLOUR_TOPIC = "/" + COLOR


def run_ready_for_live(runs_dir: Path, run: str) -> Path:
    run_dir = runs_dir / run
    if not run_dir.is_dir():
        runs = sorted(folder.name for folder in runs_dir.iterdir() if folder.is_dir()) if runs_dir.is_dir() else []
        raise SystemExit(f"no run '{run}': {run_dir} does not exist. Runs in {runs_dir}: {', '.join(runs) or 'none'}")
    what_to_do_without = {
        WEIGHTS_OF_A_RUN: f"the run has no weights. Fetch it from Colab: detect/colab/colab.sh fetch {run}",
        CONFIG_OF_A_RUN: f"the run's own copy of its settings is gone. Fetch the run again: "
                         f"detect/colab/colab.sh fetch {run}",
        CUT_OFF_OF_A_RUN: f"the run was never scored, so it has no cut-off. From detect/: "
                          f"uv run python -m detect.score --run {run_dir}",
    }
    for file, what_to_do in what_to_do_without.items():
        if not (run_dir / file).is_file():
            raise SystemExit(f"{run_dir / file} is missing: {what_to_do}")
    return run_dir


def detector_of_the_run(run_dir: Path, device: str | int):
    from detect.detector import detector_of_a_run
    return detector_of_a_run(run_dir, device)


def started(cfg: dict, run: str, load_detector=detector_of_the_run):
    run_dir = run_ready_for_live(from_repo(cfg["runs_dir"]), run)
    pub = Publisher(cfg["publish"]["host"], cfg["publish"]["port"], cfg["queue_messages"])
    try:
        detect = load_detector(run_dir, cfg["device"])
    except BaseException:
        pub.close()
        raise
    sub = Subscriber([cfg["feed_address"]], [CAMERA_TOPIC_PREFIX], cfg["queue_messages"])
    return run_dir, pub, detect, sub


def everything_waiting(sub: Subscriber, wait_for_the_first_ms: int) -> list:
    messages = []
    message = sub.recv(wait_for_the_first_ms)
    while message is not None:
        messages.append(message)
        message = sub.recv(timeout_ms=0)
    return messages


def newest_colour_image_of_each_camera(messages: list) -> tuple[dict[str, tuple], Counter]:
    newest, skipped = {}, Counter()
    for topic, fields, data in messages:
        if not topic.endswith(END_OF_A_COLOUR_TOPIC):
            continue
        camera, _ = camera_and_kind(topic)
        skipped[camera] += camera in newest
        newest[camera] = topic, fields, data
    return newest, skipped


@dataclass
class SinceLastReport:
    frames_boxed: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    model_call_s: list[float] = field(default_factory=list)
    images_in_a_call: list[int] = field(default_factory=list)

    def cameras(self) -> list[str]:
        return sorted(set(self.frames_boxed) | set(self.skipped))

    def emptied_for_the_same_cameras(self) -> "SinceLastReport":
        return SinceLastReport(frames_boxed=Counter(dict.fromkeys(self.cameras(), 0)))


def detect_on_what_is_waiting(sub: Subscriber, pub: Publisher, detect, since: SinceLastReport,
                              wait_for_the_first_ms: int) -> None:
    newest, skipped = newest_colour_image_of_each_camera(everything_waiting(sub, wait_for_the_first_ms))
    since.skipped.update(skipped)
    if not newest:
        return
    cameras = sorted(newest)
    images = [image_of(*newest[camera]) for camera in cameras]
    call_started = time.perf_counter()
    detections_per_image = detect(images)
    since.model_call_s.append(time.perf_counter() - call_started)
    since.images_in_a_call.append(len(images))
    for camera, detections in zip(cameras, detections_per_image, strict=True):
        _, fields_of_the_image, _ = newest[camera]
        found = [(d.class_name, d.box, d.confidence) for d in detections]
        pub.send(*detections_message(camera, stamp_of(fields_of_the_image), fields_of_the_image["header"]["frame_id"],
                                     found))
        since.frames_boxed[camera] += 1


def report_line(since: SinceLastReport, seconds: float, feed_address: str) -> str:
    if not since.model_call_s:
        return f"detect: waiting for colour images from {feed_address} (sim.run should be running)"
    per_camera = ", ".join(f"{camera} {since.frames_boxed[camera] / seconds:.1f} fps ({since.skipped[camera]} skipped)"
                           for camera in since.cameras())
    return (f"detect: {per_camera}; model call {1000 * statistics.median(since.model_call_s):.0f} ms "
            f"for {statistics.median(since.images_in_a_call):g} images (medians)")


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained model on the colour images of the live feed and publish its boxes, one message "
                    "per camera and frame. A frame the model has no time for is skipped, never queued.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", help="run to use instead of the one named in the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = args.run or cfg["run"]

    run_dir, pub, detect, sub = started(cfg, run)
    of_the_run = load_config(run_dir / CONFIG_OF_A_RUN)
    print(f"detect: run '{run}' ({', '.join(of_the_run['class_names'])}; image size {of_the_run['image_size']}, "
          f"cut-off {picked_cut_off(run_dir)}), frames from {cfg['feed_address']} -> boxes on {pub.address}  "
          f"(Ctrl+C to stop)")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    frames_boxed = 0
    since, last_report = SinceLastReport(), time.perf_counter()
    try:
        while True:
            until_the_next_report_ms = max(0, round(1000 * (last_report + cfg["report_every_s"] - time.perf_counter())))
            detect_on_what_is_waiting(sub, pub, detect, since, until_the_next_report_ms)
            now = time.perf_counter()
            if now - last_report >= cfg["report_every_s"]:
                print(report_line(since, now - last_report, cfg["feed_address"]))
                frames_boxed += sum(since.frames_boxed.values())
                since, last_report = since.emptied_for_the_same_cameras(), now
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sub.close()
        pub.close()
        print(f"detect: stopped after boxing {frames_boxed + sum(since.frames_boxed.values())} frames")


if __name__ == "__main__":
    main()
