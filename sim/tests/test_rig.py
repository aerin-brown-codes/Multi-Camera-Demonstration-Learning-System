from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.rig import OPENCV_TO_MUJOCO_CAMERA_AXES, Camera, Profile, Rig, add_cameras, load_rig, look_at

RIG_YAML = Path(__file__).parents[1] / "configs" / "rig.yaml"

OFF_CENTRE_NON_SQUARE = Profile(width=848, height=480, fx=615.0, fy=600.0, cx=400.0, cy=260.0,
                                depth_min_m=0.2, depth_max_m=3.0, depth_scale=0.001)

DOT_SCENE = """
<mujoco>
  <visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
  <statistic extent="1" center="0 0 0.75"/>
  <asset><material name="glow" rgba="1 1 1 1" emission="1"/></asset>
  <worldbody><body name="dot" mocap="true"><geom type="sphere" size="0.002" material="glow"/></body></worldbody>
</mujoco>"""


def project(cam: Camera, p_world) -> np.ndarray:
    p_cam = cam.R_world_cam.T @ (np.asarray(p_world) - cam.position)
    return (cam.profile.K @ (p_cam / p_cam[2]))[:2]


def test_look_at_axes():
    pos, target = np.array([0.5, -0.4, 1.3]), np.array([0.0, 0.1, 0.75])
    R = look_at(pos, target)
    right, down, forward = R.T
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1)
    assert np.allclose(forward, (target - pos) / np.linalg.norm(target - pos))
    assert np.isclose(right[2], 0)
    assert down[2] < 0


def test_look_at_straight_down():
    R = look_at([0, 0, 2], [0, 0, 0.75])
    assert np.all(np.isfinite(R))
    assert np.allclose(R[:, 2], [0, 0, -1])
    assert np.allclose(R.T @ R, np.eye(3))


def test_load_shipped_rig():
    rig = load_rig(RIG_YAML)
    assert rig.fps == 30
    assert [c.name for c in rig.cameras] == ["cam0", "cam1", "cam2", "cam3"]
    p = rig.cameras[0].profile
    assert (p.width, p.height, p.depth_scale) == (848, 480, 0.001)
    work_area_centre = [0, 0, 0.75]
    for cam in rig.cameras:
        assert np.allclose(project(cam, work_area_centre), [p.cx, p.cy], atol=1e-6)


def test_mujoco_pose_matches():
    rig = load_rig(RIG_YAML)
    spec = mujoco.MjSpec()
    add_cameras(spec, rig)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for i, cam in enumerate(rig.cameras):
        assert np.allclose(d.cam_xpos[i], cam.position)
        assert np.allclose(d.cam_xmat[i].reshape(3, 3), cam.R_world_cam * OPENCV_TO_MUJOCO_CAMERA_AXES, atol=1e-9)


@pytest.mark.parametrize("profile", [OFF_CENTRE_NON_SQUARE, load_rig(RIG_YAML).cameras[0].profile],
                         ids=["off_centre_non_square", "d435"])
def test_rendered_points_land_where_opencv_predicts(profile):
    cam = Camera("c", profile, np.array([0.3, 0.45, 1.2]), look_at([0.3, 0.45, 1.2], [0, 0, 0.75]))
    spec = mujoco.MjSpec.from_string(DOT_SCENE)
    add_cameras(spec, Rig(fps=30, cameras=[cam]))
    m = spec.compile()
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, profile.height, profile.width)
    vv, uu = np.mgrid[0:profile.height, 0:profile.width]

    rng = np.random.default_rng(0)
    errors = []
    for p in rng.uniform([-0.25, -0.2, 0.7], [0.25, 0.2, 0.9], (20, 3)):
        d.mocap_pos[0] = p
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="c")
        img = r.render()[:, :, 0].astype(float)
        assert img.sum() > 0, "dot not visible"
        centroid = np.array([(uu * img).sum(), (vv * img).sum()]) / img.sum()
        errors.append(centroid - project(cam, p))
    r.close()
    errors = np.array(errors)
    assert np.abs(errors).max() < 0.3, (
        f"max pixel error {np.abs(errors).max():.3f}: check opencv_to_mujoco_principal_pixel, "
        "MuJoCo's principal offset has both signs flipped relative to OpenCV")
