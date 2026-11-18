
"""Unit tests for reprojection error calculations."""

import cv2
import numpy as np
import pytest

from sim.reprojection import reprojection_error


def test_zero_error():
    # Four known 3D points
    points = np.array([
        [0, 0, 0],
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0.1, 0.1, 0],
    ], dtype=float)

    K = np.array([
        [610, 0, 423.5],
        [0, 610, 239.5],
        [0, 0, 1],
    ], dtype=float)

    rvec = np.zeros(3)
    tvec = np.array([0, 0, 1.0])

    # Generate perfect image observations
    image, _ = cv2.projectPoints(
        points, rvec, tvec, K, np.zeros(5)
    )

    result = reprojection_error(
        points, image, K, rvec, tvec
    )

    assert result.rmse_px == pytest.approx(0, abs=1e-9)
    assert result.n_corners == 4


def test_known_error():
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=float)

    K = np.eye(3)
    rvec = np.zeros(3)
    tvec = np.array([0, 0, 1.0])

    projected, _ = cv2.projectPoints(
        points, rvec, tvec, K, np.zeros(5)
    )

    # Add a known 5-pixel error to one corner
    observed = projected.reshape(-1, 2) + np.array([
        [3, 4],
        [0, 0],
    ])

    result = reprojection_error(
        points, observed, K, rvec, tvec
    )

    assert result.mean_px == pytest.approx(2.5)
    assert result.rmse_px == pytest.approx(np.sqrt(12.5))
    assert result.max_px == pytest.approx(5)


def test_invalid_input():
    with pytest.raises(ValueError):
        reprojection_error(
            np.zeros((2, 3)),
            np.zeros((1, 2)),
            np.eye(3),
            np.zeros(3),
            np.ones(3),
        )
