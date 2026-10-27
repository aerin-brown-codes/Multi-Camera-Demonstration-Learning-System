
"""Test camera relative pose estimation in MuJoCo."""

from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.board import load_board
from sim.calibration import detect_board_pose, relative_pose
from sim.render import RigRenderer
from sim.rig import load_rig
from sim.world import build_model


CONFIG_DIR = Path(__file__).parent / "configs"


def main():
    # 1. Load simulation configuration
    config = yaml.safe_load(
        (CONFIG_DIR / "sim.yaml").read_text()
    )

    rig = load_rig(CONFIG_DIR / config["rig"])
    board = load_board()

    # 2. Create simulation
    model = build_model(config["scene"], rig)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = RigRenderer(model, rig)

    try:
        # 3. Capture one frame from every camera
        frames = renderer.render(data)
    finally:
        renderer.close()

    # 4. Detect board pose independently in each image
    detected = {}

    for camera in rig.cameras:
        rgb, depth = frames[camera.name]

        T = detect_board_pose(
            rgb,
            camera.profile.K,
            board,
        )

        if T is None:
            print(f"{camera.name}: board NOT detected")
        else:
            print(f"{camera.name}: board detected")
            detected[camera.name] = T

    # 5. Select cam0 as reference camera
    reference = rig.cameras[0]

    if reference.name not in detected:
        print("Reference camera detection failed.")
        return

    print("\nRelative camera poses (reference: cam0)")

    # 6. Calculate relative position of other cameras
    for camera in rig.cameras[1:]:
        if camera.name not in detected:
            print(f"{camera.name}: skipped")
            continue

        T_estimated = relative_pose(
            detected[reference.name],
            detected[camera.name],
        )

        # Ground truth from simulated camera positions
        T_ground_truth = (
            np.linalg.inv(reference.T_world_cam)
            @ camera.T_world_cam
        )

        # Translation error, expressed in metres
        position_error = np.linalg.norm(
            T_estimated[:3, 3]
            - T_ground_truth[:3, 3]
        )

        # Rotation error, expressed in degrees
        R_error = (
            T_estimated[:3, :3]
            @ T_ground_truth[:3, :3].T
        )

        cos_angle = np.clip(
            (np.trace(R_error) - 1) / 2,
            -1.0,
            1.0,
        )

        angle_error = np.degrees(np.arccos(cos_angle))

        print(f"\n{camera.name} relative to {reference.name}:")
        print("Estimated translation (m):")
        print(T_estimated[:3, 3])

        print("Ground truth translation (m):")
        print(T_ground_truth[:3, 3])

        print(f"Position error: {position_error * 1000:.2f} mm")
        print(f"Rotation error: {angle_error:.2f} degrees")


if __name__ == "__main__":
    main()
