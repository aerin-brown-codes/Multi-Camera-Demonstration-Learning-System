from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

CONFIG = Path(__file__).parent / "configs" / "board.yaml"
ASSETS = Path(__file__).parent / "assets"
BOARD_AXES_IN_BOX_FRAME = np.array([1, -1, -1])


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float
    dictionary: str
    png_px_per_mm: int

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    @property
    def texture_name(self) -> str:
        return f"charuco_{self.squares_x}x{self.squares_y}_{round(self.square_m * 1000)}mm.png"

    def cv_board(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m, dictionary)


def load_board(path=CONFIG) -> BoardSpec:
    return BoardSpec(**yaml.safe_load(Path(path).read_text()))


def T_world_board(model: mujoco.MjModel, data: mujoco.MjData, geom: str = "board") -> np.ndarray:
    gid = model.geom(geom).id
    R_world_box = data.geom_xmat[gid].reshape(3, 3)
    half_x, half_y, half_z = model.geom_size[gid]
    texture_top_left_in_box = np.array([-half_x, half_y, half_z])
    T = np.eye(4)
    T[:3, :3] = R_world_box * BOARD_AXES_IN_BOX_FRAME
    T[:3, 3] = data.geom_xpos[gid] + R_world_box @ texture_top_left_in_box
    return T


def main():
    spec = load_board()
    w, h = (round(s * 1000 * spec.png_px_per_mm) for s in spec.size_m)
    img = spec.cv_board().generateImage((w, h), marginSize=0)
    out = ASSETS / spec.texture_name
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    print(f"wrote {out} ({w}x{h} px, {spec.size_m[0]:.3f} x {spec.size_m[1]:.3f} m)")


if __name__ == "__main__":
    main()
