import mujoco
import numpy as np

from sim.rig import Profile, Rig


def to_depth_units(depth_m: np.ndarray, profile: Profile) -> np.ndarray:
    in_range = (depth_m >= profile.depth_min_m) & (depth_m <= profile.depth_max_m)
    return np.where(in_range, np.round(depth_m / profile.depth_scale), 0).astype(np.uint16)


def disable_multisampling_for_exact_depth(model: mujoco.MjModel) -> None:
    model.vis.quality.offsamples = 0


class RigRenderer:
    def __init__(self, model: mujoco.MjModel, rig: Rig):
        self.rig = rig
        disable_multisampling_for_exact_depth(model)
        self._renderers = {
            size: mujoco.Renderer(model, size[1], size[0])
            for size in {(c.profile.width, c.profile.height) for c in rig.cameras}
        }
        extent = model.stat.extent
        self._near_m = np.float32(model.vis.map.znear * extent)
        self._far_m = np.float32(model.vis.map.zfar * extent)

    def _reversed_z_to_metres(self, z: np.ndarray) -> np.ndarray:
        n, f = self._near_m, self._far_m
        return f * n / ((f - n) * z + n)

    def _depth_buffer_of_last_render_m(self, r: mujoco.Renderer) -> np.ndarray:
        z_bottom_row_first = np.empty((r.height, r.width), np.float32)
        mujoco.mjr_readPixels(None, z_bottom_row_first, r._rect, r._mjr_context)
        return np.flipud(self._reversed_z_to_metres(z_bottom_row_first))

    def render(self, data: mujoco.MjData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        frames = {}
        for cam in self.rig.cameras:
            r = self._renderers[cam.profile.width, cam.profile.height]
            r.update_scene(data, camera=cam.name)
            rgb = r.render()
            frames[cam.name] = rgb, to_depth_units(self._depth_buffer_of_last_render_m(r), cam.profile)
        return frames

    def close(self):
        for r in self._renderers.values():
            r.close()
