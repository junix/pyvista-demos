from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from . import SCENES

PALETTE = ("#54d6c6", "#ffd166", "#ff6f91", "#7b9cff", "#b889ff")


def new_plotter(title: str, subtitle: str) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.set_background("#07111f")
    plotter.add_text(title, position=(55, 925), font_size=22, color="#edf8ff", font="arial")
    plotter.add_text(subtitle, position=(58, 886), font_size=10, color="#82a0b5", font="arial")
    return plotter


def spectral_terrain() -> pv.Plotter:
    plotter = new_plotter("SPECTRAL TERRAIN", "SYNTHETIC ELEVATION / CONTOUR FREQUENCY")
    x = np.linspace(-5.5, 5.5, 180)
    y = np.linspace(-4.1, 4.1, 150)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = 0.7 * np.sin(xx * 1.2) * np.cos(yy * 1.45) + 0.28 * np.sin((xx + yy) * 3.1)
    zz += 0.18 * np.cos(np.sqrt(xx**2 + yy**2) * 5.0)
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["elevation"] = zz.ravel(order="F")
    plotter.add_mesh(
        grid,
        scalars="elevation",
        cmap=["#172d54", "#346cb3", "#54d6c6", "#ffd166", "#ff6f91"],
        smooth_shading=True,
        specular=0.45,
        metallic=0.18,
        show_scalar_bar=False,
    )
    contours = grid.contour(isosurfaces=16, scalars="elevation")
    plotter.add_mesh(contours, color="#d8f7ff", line_width=1.2, opacity=0.24)
    plotter.camera_position = [(11.8, -12.5, 9.5), (0, 0, 0), (0, 0, 1)]
    return plotter


def vortex_streamlines() -> pv.Plotter:
    plotter = new_plotter("VORTEX CHAMBER", "VECTOR FIELD / SEEDED STREAMLINES")
    grid = pv.ImageData(dimensions=(44, 44, 34), spacing=(0.22, 0.22, 0.22), origin=(-4.7, -4.7, -3.6))
    points = grid.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radial = np.sqrt(x**2 + y**2) + 0.25
    vectors = np.column_stack((-y / radial, x / radial, 0.32 * np.sin(radial * 1.8) - 0.08 * z))
    vectors *= np.exp(-0.035 * (x**2 + y**2 + z**2))[:, None] + 0.28
    grid["vectors"] = vectors
    grid["speed"] = np.linalg.norm(vectors, axis=1)
    seeds = pv.Disc(inner=0.3, outer=3.7, r_res=7, c_res=32)
    stream = grid.streamlines_from_source(
        seeds,
        vectors="vectors",
        max_step_length=0.12,
        max_length=18,
        integration_direction="both",
    )
    tubes = stream.tube(radius=0.025, scalars="speed", radius_factor=3.2)
    plotter.add_mesh(tubes, scalars="speed", cmap=["#54d6c6", "#7b9cff", "#ff6f91", "#ffd166"], opacity=0.88, show_scalar_bar=False)
    shell = pv.Sphere(radius=4.25, theta_resolution=64, phi_resolution=40)
    plotter.add_mesh(shell, style="wireframe", color="#83a3ba", opacity=0.09, line_width=1)
    plotter.camera_position = [(10.5, -11.8, 7.7), (0, 0, 0), (0, 0, 1)]
    return plotter


def wave_isosurface() -> pv.Plotter:
    plotter = new_plotter("WAVE ISOSURFACES", "THREE-DIMENSIONAL INTERFERENCE FIELD")
    grid = pv.ImageData(dimensions=(88, 88, 88), spacing=(0.1, 0.1, 0.1), origin=(-4.35, -4.35, -4.35))
    x, y, z = grid.points.T
    r1 = np.sqrt((x + 1.8) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.8) ** 2 + y**2 + z**2)
    field = np.sin(3.2 * r1) / (r1 + 0.45) + np.sin(3.2 * r2 + 0.7) / (r2 + 0.45)
    grid["field"] = field
    for value, color, opacity in ((-0.45, PALETTE[3], 0.23), (0.0, PALETTE[0], 0.18), (0.45, PALETTE[2], 0.28)):
        surface = grid.contour([value], scalars="field")
        plotter.add_mesh(surface, color=color, opacity=opacity, smooth_shading=True, specular=0.65)
    for center, color in (((-1.8, 0, 0), PALETTE[1]), ((1.8, 0, 0), PALETTE[4])):
        plotter.add_mesh(pv.Sphere(radius=0.24, center=center), color=color, emissive=True)
    plotter.camera_position = [(11.7, -10.8, 8.4), (0, 0, 0), (0, 0, 1)]
    return plotter


BUILDERS = {
    "spectral-terrain": spectral_terrain,
    "vortex-streamlines": vortex_streamlines,
    "wave-isosurface": wave_isosurface,
}


def validate_png(path: Path) -> dict[str, object]:
    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A").histogram()
    pixels = image.width * image.height
    transparent = alpha[0]
    visible = pixels - sum(alpha[:16])
    colors = image.getcolors(maxcolors=pixels) or []
    colorful = sum(count for count, rgba in colors if rgba[3] > 16 and max(rgba[:3]) - min(rgba[:3]) > 24)
    if transparent < pixels * 0.06 or visible < pixels * 0.035 or colorful < 2500:
        raise RuntimeError(f"{path.name}: weak RGBA content t={transparent} v={visible} c={colorful}")
    return {
        "scene": path.stem.removesuffix("-transparent"),
        "size": image.size,
        "transparent_pct": round(100 * transparent / pixels, 1),
        "visible_pct": round(100 * visible / pixels, 1),
        "colorful": colorful,
    }


def render(scene: str, output: Path) -> dict[str, object]:
    plotter = BUILDERS[scene]()
    plotter.show(auto_close=False)
    plotter.screenshot(output, transparent_background=True)
    plotter.close()
    return validate_png(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = (args.scene,) if args.scene else SCENES
    report = [render(scene, args.out / f"{scene}-transparent.png") for scene in chosen]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
