
"""Estimate camera poses using a ChArUco calibration board."""

import cv2
import numpy as np

from sim.board import BoardSpec


def detect_board_pose(
    rgb: np.ndarray,
    K: np.ndarray,
    spec: BoardSpec,
) -> np.ndarray | None:
    """
    Detect the ChArUco board in an RGB image.

    Returns a 4x4 transformation matrix T_camera_board,
    or None if detection fails.
    """

    board = spec.cv_board()
    detector = cv2.aruco.CharucoDetector(board)

    # Convert image to grayscale for marker detection
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Find ChArUco corners and their IDs
    corners, ids, _, _ = detector.detectBoard(gray)

    if ids is None or len(ids) < 4:
        return None

    # Match detected image corners with 3D board points
    object_points = (
        board.getChessboardCorners()[ids.flatten()]
        .astype(np.float32)
    )
    image_points = corners.reshape(-1, 2).astype(np.float32)

    # Estimate board rotation and translation
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(K, dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T


def relative_pose(
    T_camA_board: np.ndarray,
    T_camB_board: np.ndarray,
) -> np.ndarray:
    """
    Calculate camera B's pose relative to camera A.

    Returns a 4x4 transformation T_camA_camB.
    """

    return T_camA_board @ np.linalg.inv(T_camB_board)
