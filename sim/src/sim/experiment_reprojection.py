
"""Compare ChArUco reprojection errors across camera distances in MuJoCo."""

import csv
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.board import load_board
from sim.render import RigRenderer
from sim.reprojection import measure_board_reprojection
from sim.rig import Rig, load_rig, look_at
from sim.world import build_model

CONFIG_DIR = Path(__file__).parent / "configs"
OUTPUT_DIR = Path.cwd() / "reprojection_experiment_results"
TARGET = np.array([0.0, 0.0, 0.75])

SCALES = {
    "close": 0.85,
    "baseline": 1.0,
    "far": 1.15,
}


def scaled_rig(base_rig, scale):
    """Move all cameras closer or farther from the target."""
    cameras = []

    for camera in base_rig.cameras:
        position = TARGET + scale * (camera.position - TARGET)

        cameras.append(
            replace(
                camera,
                position=position,
                R_world_cam=look_at(position, TARGET),
            )
        )

    return Rig(fps=base_rig.fps, cameras=cameras)


def main():
    config = yaml.safe_load(
        (CONFIG_DIR / "sim.yaml").read_text()
    )
    base_rig = load_rig(CONFIG_DIR / config["rig"])
    board = load_board()

    rows = []
    OUTPUT_DIR.mkdir(exist_ok=True)

    for condition, scale in SCALES.items():
        rig = scaled_rig(base_rig, scale)

        # Build the scene with the new camera positions
        model = build_model(config["scene"], rig)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = RigRenderer(model, rig)

        try:
            frames = renderer.render(data)
        finally:
            renderer.close()

        print(f"\n{condition.upper()} (distance scale {scale:.2f}x)")

        for camera in rig.cameras:
            rgb, _depth = frames[camera.name]

            result = measure_board_reprojection(
                rgb,
                camera.profile.K,
                board,
            )

            row = {
                "condition": condition,
                "distance_scale": scale,
                "camera": camera.name,
                "rmse_px": "" if result is None else round(result.rmse_px, 6),
                "mean_px": "" if result is None else round(result.mean_px, 6),
                "median_px": "" if result is None else round(result.median_px, 6),
                "max_px": "" if result is None else round(result.max_px, 6),
                "n_corners": 0 if result is None else result.n_corners,
                "detected": result is not None,
            }

            rows.append(row)

            if result is None:
                print(f"  {camera.name}: board NOT detected")
            else:
                print(
                    f"  {camera.name}: RMSE = {result.rmse_px:.3f} px; "
                    f"corners = {result.n_corners}"
                )

    # Export results as CSV
    csv_path = OUTPUT_DIR / "reprojection_by_camera.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Generate comparison graph
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for camera in base_rig.cameras:
        subset = [
            r for r in rows if r["camera"] == camera.name
        ]

        x = [
            float(r["distance_scale"])
            for r in subset if r["detected"]
        ]
        y = [
            float(r["rmse_px"])
            for r in subset if r["detected"]
        ]

        ax.plot(x, y, marker="o", label=camera.name)

    ax.set_xlabel("Distance relative to baseline (×)")
    ax.set_ylabel("Reprojection RMSE (pixels)")
    ax.set_title("ChArUco reprojection error vs camera distance")
    ax.set_xticks(list(SCALES.values()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "reprojection_vs_distance.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(
        "Interpretation: these are single-frame simulation measurements, "
        "not repeated trials or physical-camera validation."
    )


if __name__ == "__main__":
    main()
