import argparse
import signal
import sys
import time
from pathlib import Path

import mujoco
import yaml

from sim.feed import Publisher
from sim.poses import Poser, held_pose_index, load_scene_poses
from sim.render import RigRenderer
from sim.rig import Rig, load_rig
from sim.world import build_model

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "sim.yaml"


def make_truth(model: mujoco.MjModel, data: mujoco.MjData, rig: Rig, frame: int) -> dict:
    bodies = {}
    for i in range(model.nbody):
        name = model.body(i).name
        if name and name != "world":
            bodies[name] = {"pos": data.xpos[i].tolist(), "quat_wxyz": data.xquat[i].tolist()}
    cameras = {cam.name: cam.T_world_cam.tolist() for cam in rig.cameras}
    return {"frame": frame, "t": data.time, "cameras": cameras, "bodies": bodies}


def frame_advancer(cfg: dict, config_dir: Path, scene: str, model: mujoco.MjModel, data: mujoco.MjData, fps: float):
    scene_config = config_dir / "scenes" / f"{scene}.yaml"
    scene_poses = load_scene_poses(scene_config) if scene_config.exists() else None
    if scene_poses is None or not scene_poses.poses:
        def step_physics_to(frame: int) -> None:
            while data.time < frame / fps:
                mujoco.mj_step(model, data)
        return step_physics_to, "physics"

    poses = scene_poses.poses
    poser = Poser(model, scene_poses)
    poser.require_no_overlaps(data, poses, cfg["allowed_overlap_m"])

    def place_held_pose_at(frame: int) -> None:
        poser.place(data, poses[held_pose_index(frame, fps, cfg["pose_hold_s"], len(poses))])
        data.time = frame / fps
    return place_held_pose_at, f"{len(poses)} poses held {cfg['pose_hold_s']:g} s each, no physics"


def main():
    parser = argparse.ArgumentParser(
        description="Run the sim and stream every rig camera live. If rendering can't keep up, "
                    "sim time slows down instead of dropping frames.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", help="scene to load instead of the one named in the config")
    args = parser.parse_args()
    config_path = args.config
    cfg = yaml.safe_load(config_path.read_text())
    scene = args.scene or cfg["scene"]

    rig = load_rig(config_path.parent / cfg["rig"])
    model = build_model(scene, rig)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    advance_to, mode = frame_advancer(cfg, config_path.parent, scene, model, data, rig.fps)
    renderer = RigRenderer(model, rig)
    feed = cfg["feed"]
    pub = Publisher(feed["host"], feed["port"], feed["queue_messages"])
    print(f"sim: scene '{scene}' ({mode}), {len(rig.cameras)} cameras "
          f"({', '.join(c.name for c in rig.cameras)}) at {rig.fps:g} fps -> {pub.address}  (Ctrl+C to stop)")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    frame = 0
    wall_time_at_sim_zero = time.perf_counter()
    report_wall, report_sim = wall_time_at_sim_zero, 0.0
    try:
        while True:
            advance_to(frame)
            frames = renderer.render(data)
            for cam in rig.cameras:
                pub.send_frame(cam, frame, data.time, *frames[cam.name])
            pub.send_truth(make_truth(model, data, rig, frame))
            frame += 1

            now = time.perf_counter()
            behind_s = now - (wall_time_at_sim_zero + frame / rig.fps)
            if behind_s < 0:
                time.sleep(-behind_s)
            elif behind_s > 1 / rig.fps:
                wall_time_at_sim_zero += behind_s

            if now - report_wall >= cfg["speed_report_s"]:
                speed = (data.time - report_sim) / (now - report_wall)
                if speed < cfg["speed_warn_below"]:
                    print(f"sim: running at {speed:.2f}x real time ({speed * rig.fps:.1f} fps)")
                report_wall, report_sim = now, data.time
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()
        pub.close()
        print(f"sim: stopped after {frame} frames ({data.time:.1f} s sim time)")


if __name__ == "__main__":
    main()
