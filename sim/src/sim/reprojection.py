
"""Measure ChArUco calibration reprojection error in pixels."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReprojectionResult:
    rmse_px: float
    mean_px: float
    median_px: float
    max_px: float
    n_corners: int
    residuals_px: np.ndarray


def reprojection_error(
    object_points,
    image_points,
    K,
    rvec,
    tvec,
    dist_coeffs=None,
):
    """Compare detected 2D points with projected 3D points."""

    object_points = np.asarray(
        object_points, dtype=np.float64
    ).reshape(-1, 3)

    image_points = np.asarray(
        image_points, dtype=np.float64
    ).reshape(-1, 2)

    if len(object_points) == 0 or len(object_points) != len(image_points):
        raise ValueError("Invalid number of points")

    K = np.asarray(K, dtype=np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    # Project known 3D corners into the camera image
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist_coeffs,
    )

    projected = projected.reshape(-1, 2)

    # Euclidean distance in pixels for each corner
    residuals = np.linalg.norm(
        image_points - projected,
        axis=1,
    )

    # Root mean squared reprojection error
    rmse = np.sqrt(np.mean(residuals ** 2))

    return ReprojectionResult(
        rmse_px=float(rmse),
        mean_px=float(np.mean(residuals)),
        median_px=float(np.median(residuals)),
        max_px=float(np.max(residuals)),
        n_corners=len(residuals),
        residuals_px=residuals,
    )


def measure_board_reprojection(
    rgb,
    K,
    spec,
    dist_coeffs=None,
):
    """Detect a ChArUco board and measure reprojection error."""

    board = spec.cv_board()

    # Convert rendered RGB image to grayscale
    gray = cv2.cvtColor(
        rgb, cv2.COLOR_RGB2GRAY
    )

    # Detect ChArUco corners
    detector = cv2.aruco.CharucoDetector(board)

    corners, ids, _, _ = detector.detectBoard(gray)

    if ids is None or len(ids) < 4:
        return None

    # Match detected corners to their 3D positions
    object_points = board.getChessboardCorners()[
        ids.flatten()
    ].astype(np.float64)

    image_points = corners.reshape(
        -1, 2
    ).astype(np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    # Estimate board orientation and position
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(K, dtype=np.float64),
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    return reprojection_error(
        object_points,
        image_points,
        K,
        rvec,
        tvec,
        dist_coeffs,
    )
