import argparse
import signal
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from feed.detections import detections_message, positions_message
from feed.frames import Frame, FramePairer
from feed.messages import CAMERA_TOPIC_PREFIX, COLOR, METRES_PER_DEPTH_UNIT, camera_and_kind, color_optical_frame
from feed.pipe import Publisher, Subscriber
from feed.transforms import TABLE_FRAME, TF_STATIC_TOPIC, static_transforms_of

from detect.config import CONFIGS, from_repo, load_config
from detect.lift import position_or_none_without_depth
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
    sub = Subscriber([cfg["feed_address"]], [CAMERA_TOPIC_PREFIX, TF_STATIC_TOPIC], cfg["queue_messages"])
    return run_dir, pub, detect, sub


def lift_as_configured(cfg: dict):
    return partial(position_or_none_without_depth, metres_per_depth_unit=METRES_PER_DEPTH_UNIT, **cfg["lift"])


def everything_waiting(sub: Subscriber, wait_for_the_first_ms: int) -> list:
    messages = []
    message = sub.recv(wait_for_the_first_ms)
    while message is not None:
        messages.append(message)
        message = sub.recv(timeout_ms=0)
    return messages


@dataclass
class SeenOnTheFeed:
    pairer: FramePairer = field(default_factory=FramePairer)
    T_table_cam_of_frame_id: dict[str, np.ndarray] = field(default_factory=dict)


def newest_complete_frame_of_each_camera(messages: list, seen: SeenOnTheFeed) -> tuple[dict[str, Frame], Counter]:
    newest, skipped = {}, Counter()
    for topic, fields, data in messages:
        if topic == TF_STATIC_TOPIC:
            seen.T_table_cam_of_frame_id.update(static_transforms_of(fields, TABLE_FRAME))
            continue
        frame = seen.pairer.add(topic, fields, data)
        if frame is not None:
            skipped[frame.camera] += frame.camera in newest
            newest[frame.camera] = frame
    return newest, skipped


def cameras_of_the_colour_images_in(messages: list) -> list[str]:
    return [camera_and_kind(topic)[0] for topic, _, _ in messages if topic.endswith(END_OF_A_COLOUR_TOPIC)]


@dataclass
class SinceLastReport:
    frames_boxed: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    colour_images: Counter = field(default_factory=Counter)
    model_call_s: list[float] = field(default_factory=list)
    images_in_a_call: list[int] = field(default_factory=list)
    boxes: int = 0
    boxes_lifted: int = 0
    cameras_without_a_pose: set[str] = field(default_factory=set)

    def cameras(self) -> list[str]:
        return sorted(set(self.frames_boxed) | set(self.skipped))

    def emptied_for_the_same_cameras(self) -> "SinceLastReport":
        return SinceLastReport(frames_boxed=Counter(dict.fromkeys(self.cameras(), 0)))


def detect_on_what_is_waiting(sub: Subscriber, pub: Publisher, detect, lift, seen: SeenOnTheFeed,
                              since: SinceLastReport, wait_for_the_first_ms: int) -> None:
    messages = everything_waiting(sub, wait_for_the_first_ms)
    since.colour_images.update(cameras_of_the_colour_images_in(messages))
    newest, skipped = newest_complete_frame_of_each_camera(messages, seen)
    since.skipped.update(skipped)
    if not newest:
        return
    cameras = sorted(newest)
    call_started = time.perf_counter()
    detections_per_image = detect([newest[camera].rgb for camera in cameras])
    since.model_call_s.append(time.perf_counter() - call_started)
    since.images_in_a_call.append(len(cameras))
    for camera, detections in zip(cameras, detections_per_image, strict=True):
        frame, frame_id = newest[camera], color_optical_frame(camera)
        found = [(d.class_name, d.box, d.confidence) for d in detections]
        pub.send(*detections_message(camera, frame.stamp, frame_id, found))
        since.frames_boxed[camera] += 1
        since.boxes += len(detections)
        T_table_cam = seen.T_table_cam_of_frame_id.get(frame_id)
        if T_table_cam is None:
            since.cameras_without_a_pose.add(camera)
            continue
        lifted = []
        for place_in_the_boxes_message, d in enumerate(detections):
            point = lift(d.box, frame.depth, K=frame.K, T_table_cam=T_table_cam)
            if point is not None:
                lifted.append((str(place_in_the_boxes_message), d.class_name, tuple(point.tolist()), d.confidence))
        pub.send(*positions_message(camera, frame.stamp, lifted))
        since.boxes_lifted += len(lifted)


def report_line(since: SinceLastReport, seconds: float, feed_address: str) -> str:
    if not since.model_call_s:
        return f"detect: waiting for frames from {feed_address} (sim.run should be running)"
    per_camera = ", ".join(f"{camera} {since.frames_boxed[camera] / seconds:.1f} fps ({since.skipped[camera]} skipped)"
                           for camera in since.cameras())
    return (f"detect: {per_camera}; model call {1000 * statistics.median(since.model_call_s):.0f} ms "
            f"for {statistics.median(since.images_in_a_call):g} images (medians); "
            f"{since.boxes_lifted} of {since.boxes} boxes lifted")


def lines_about_what_is_missing(since: SinceLastReport) -> list[str]:
    lines = []
    without_depth = sorted(camera for camera, colour_images in since.colour_images.items()
                           if colour_images and not since.frames_boxed[camera] + since.skipped[camera])
    if without_depth:
        lines.append(f"detect: no depth images for {', '.join(without_depth)}, so no boxes")
    if since.cameras_without_a_pose:
        lines.append(f"detect: no camera pose for {', '.join(sorted(since.cameras_without_a_pose))} "
                     f"on {TF_STATIC_TOPIC}, so boxes only")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained model on the frames of the live feed and publish its boxes and where each box is "
                    "in table coordinates, one message of each per camera and frame. A frame the model has no time "
                    "for is skipped, never queued.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", help="run to use instead of the one named in the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = args.run or cfg["run"]

    run_dir, pub, detect, sub = started(cfg, run)
    of_the_run = load_config(run_dir / CONFIG_OF_A_RUN)
    print(f"detect: run '{run}' ({', '.join(of_the_run['class_names'])}; image size {of_the_run['image_size']}, "
          f"cut-off {picked_cut_off(run_dir)}), frames from {cfg['feed_address']} -> boxes and positions on "
          f"{pub.address}  "
          f"(Ctrl+C to stop)")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    frames_boxed = 0
    lift, seen = lift_as_configured(cfg), SeenOnTheFeed()
    since, last_report = SinceLastReport(), time.perf_counter()
    try:
        while True:
            until_the_next_report_ms = max(0, round(1000 * (last_report + cfg["report_every_s"] - time.perf_counter())))
            detect_on_what_is_waiting(sub, pub, detect, lift, seen, since, until_the_next_report_ms)
            now = time.perf_counter()
            if now - last_report >= cfg["report_every_s"]:
                print("\n".join([report_line(since, now - last_report, cfg["feed_address"]),
                                 *lines_about_what_is_missing(since)]))
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
