
"""Visualize ChArUco detection for all simulated cameras."""

from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from sim.board import load_board
from sim.calibration import detect_board_pose
from sim.render import RigRenderer
from sim.rig import load_rig
from sim.world import build_model


CONFIG_DIR = Path(__file__).parent / "configs"
OUTPUT_DIR = Path.cwd() / "calibration_results"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the simulation
    config = yaml.safe_load(
        (CONFIG_DIR / "sim.yaml").read_text()
    )

    rig = load_rig(CONFIG_DIR / config["rig"])
    board_spec = load_board()
    board = board_spec.cv_board()

    # Create the virtual environment
    model = build_model(config["scene"], rig)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = RigRenderer(model, rig)

    try:
        frames = renderer.render(data)
    finally:
        renderer.close()

    detector = cv2.aruco.CharucoDetector(board)

    tiles = []

    for camera in rig.cameras:
        rgb, depth = frames[camera.name]

        # OpenCV drawing uses BGR
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        # Detect ChArUco corners
        corners, ids, marker_corners, marker_ids = (
            detector.detectBoard(gray)
        )

        # Draw detected ArUco markers
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                image,
                marker_corners,
                marker_ids
            )

        
        # Draw detected ChArUco corners
        if (
            corners is not None
            and ids is not None
            and len(corners) > 0
            and len(corners) == len(ids)
        ):
            charuco_corners = np.asarray(
                corners, dtype=np.float32
            ).reshape(-1, 1, 2)

            charuco_ids = np.asarray(
                ids, dtype=np.int32
            ).reshape(-1, 1)

            cv2.aruco.drawDetectedCornersCharuco(
                image,
                charuco_corners,
                charuco_ids,
                (0, 255, 0)
            )


        # Estimate the board pose
        T = detect_board_pose(
            rgb,
            camera.profile.K,
            board_spec
        )

        if T is not None:
            status = "DETECTED"
            color = (0, 255, 0)

            # Draw the board coordinate axes
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            tvec = T[:3, 3].reshape(3, 1)

            cv2.drawFrameAxes(
                image,
                camera.profile.K.astype(np.float64),
                np.zeros(5),
                rvec,
                tvec,
                0.10
            )

        else:
            status = "NOT DETECTED"
            color = (0, 0, 255)

        # Add camera name and detection status
        cv2.putText(
            image,
            f"{camera.name}: {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        count = 0 if ids is None else len(ids)

        cv2.putText(
            image,
            f"ChArUco corners: {count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Save individual camera image
        filename = OUTPUT_DIR / f"{camera.name}.png"
        cv2.imwrite(str(filename), image)

        # Resize for 2x2 overview
        tile = cv2.resize(image, (640, 360))
        tiles.append(tile)

        print(
            f"{camera.name}: {status}, "
            f"{count} corners"
        )

    # Combine all four images into one overview
    if len(tiles) == 4:
        top = np.hstack([tiles[0], tiles[1]])
        bottom = np.hstack([tiles[2], tiles[3]])
        overview = np.vstack([top, bottom])

        output = OUTPUT_DIR / "overview.png"
        cv2.imwrite(str(output), overview)

        print(f"\nOverview saved to: {output}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
