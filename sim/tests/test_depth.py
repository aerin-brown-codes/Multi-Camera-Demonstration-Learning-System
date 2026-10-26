import mujoco
import numpy as np

from sim.render import to_depth_units

MAX_ROUNDING_ERROR_MM = 0.5


def test_depth_units_encoding(tabletop):
    profile = tabletop.rig.cameras[0].profile
    metres = np.array([0.1, 0.2, 1.2344, 1.2346, 3.0, 3.01, 50.0])
    depth = to_depth_units(metres, profile)
    assert depth.dtype == np.uint16
    assert depth.tolist() == [0, 200, 1234, 1235, 3000, 0, 0]


def test_depth_matches_raycast(tabletop):
    m, d = tabletop.model, tabletop.data
    rng = np.random.default_rng(0)
    geomid = np.zeros(1, np.int32)
    for cam in tabletop.rig.cameras:
        p = cam.profile
        depth = tabletop.frames[cam.name][1]
        K_inv = np.linalg.inv(p.K)
        errors_mm, validity_mismatch = [], 0
        for u, v in zip(rng.integers(0, p.width, 3000), rng.integers(0, p.height, 3000)):
            ray_with_unit_z_in_camera = cam.R_world_cam @ (K_inv @ [u, v, 1.0])
            z_depth_m = mujoco.mj_ray(m, d, cam.position, ray_with_unit_z_in_camera, None, 1, -1, geomid)
            expected_valid = p.depth_min_m <= z_depth_m <= p.depth_max_m
            if expected_valid != (depth[v, u] > 0):
                validity_mismatch += 1
            elif expected_valid:
                errors_mm.append((depth[v, u] * p.depth_scale - z_depth_m) * 1000)
        errors_mm = np.abs(errors_mm)
        assert len(errors_mm) > 1000
        assert np.percentile(errors_mm, 99) < 1.0, (
            f"{cam.name}: 99th pct {np.percentile(errors_mm, 99):.2f} mm (multisampling on? "
            "it makes the depth readback off-centre)")
        assert np.median(errors_mm) < MAX_ROUNDING_ERROR_MM
        assert validity_mismatch < 30, f"{cam.name}: {validity_mismatch} pixels disagree on having a reading"
