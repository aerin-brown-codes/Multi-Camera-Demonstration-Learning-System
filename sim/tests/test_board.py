import cv2
import numpy as np
import pytest

from sim.board import T_world_board, load_board

TEXTURE_FILTERING_BIAS_ALLOWANCE_MM = 3.0
TEXTURE_FILTERING_BIAS_ALLOWANCE_DEG = 0.2


def test_board_geom_matches_config(tabletop):
    m = tabletop.model
    size = 2 * m.geom_size[m.geom("board").id][:2]
    assert np.allclose(size, load_board().size_m)


@pytest.fixture(scope="module")
def detections(tabletop):
    board = load_board().cv_board()
    detector = cv2.aruco.CharucoDetector(board)
    out = []
    for cam in tabletop.rig.cameras:
        rgb = tabletop.frames[cam.name][0]
        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        assert ids is not None and len(ids) >= 12, f"{cam.name}: board not found"
        out.append((cam, corners.reshape(-1, 2), board.getChessboardCorners()[ids.flatten()]))
    return out


def test_board_corners_render_where_truth_says(tabletop, detections):
    T_world_board_true = T_world_board(tabletop.model, tabletop.data)
    for cam, detected, corners_in_board in detections:
        T_cam_board = np.linalg.inv(cam.T_world_cam) @ T_world_board_true
        p_cam = corners_in_board @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
        projected = (p_cam / p_cam[:, 2:]) @ cam.profile.K.T
        err = detected - projected[:, :2]
        assert np.abs(err.mean(0)).max() < 0.1, f"{cam.name}: corners offset by {err.mean(0)} px"
        assert np.sqrt((err**2).mean()) < 0.3, f"{cam.name}: corner rms error {np.sqrt((err**2).mean()):.2f} px"


def test_board_pose_from_each_camera_within_texture_filtering_bias(tabletop, detections):
    T_world_board_true = T_world_board(tabletop.model, tabletop.data)
    for cam, detected, corners_in_board in detections:
        ok, rvec, tvec = cv2.solvePnP(corners_in_board, detected, cam.profile.K, None)
        assert ok
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_board[:3, 3] = tvec.flatten()
        est = cam.T_world_cam @ T_cam_board

        pos_err_mm = np.linalg.norm(est[:3, 3] - T_world_board_true[:3, 3]) * 1000
        cos = (np.trace(est[:3, :3].T @ T_world_board_true[:3, :3]) - 1) / 2
        ang_err_deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        assert pos_err_mm < TEXTURE_FILTERING_BIAS_ALLOWANCE_MM, f"{cam.name}: board position off by {pos_err_mm:.2f} mm"
        assert ang_err_deg < TEXTURE_FILTERING_BIAS_ALLOWANCE_DEG, f"{cam.name}: board rotation off by {ang_err_deg:.3f} deg"
