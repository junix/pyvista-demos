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
    plotter.add_mesh(
        tubes,
        scalars="speed",
        cmap=["#54d6c6", "#7b9cff", "#ff6f91", "#ffd166"],
        opacity=0.88,
        show_scalar_bar=False,
    )
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
    for value, color, opacity in (
        (-0.45, PALETTE[3], 0.23),
        (0.0, PALETTE[0], 0.18),
        (0.45, PALETTE[2], 0.28),
    ):
        surface = grid.contour([value], scalars="field")
        plotter.add_mesh(surface, color=color, opacity=opacity, smooth_shading=True, specular=0.65)
    for center, color in (((-1.8, 0, 0), PALETTE[1]), ((1.8, 0, 0), PALETTE[4])):
        plotter.add_mesh(pv.Sphere(radius=0.24, center=center), color=color, emissive=True)
    plotter.camera_position = [(11.7, -10.8, 8.4), (0, 0, 0), (0, 0, 1)]
    return plotter


def gyroid_lattice() -> pv.Plotter:
    plotter = new_plotter("GYROID LATTICE", "TRIPLY PERIODIC MINIMAL SURFACE / POROSITY STUDY")
    grid = pv.ImageData(dimensions=(76, 76, 76), spacing=(0.11, 0.11, 0.11), origin=(-4.12, -4.12, -4.12))
    x, y, z = grid.points.T
    field = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    grid["gyroid"] = field
    for value, color, opacity in ((-0.38, PALETTE[3], 0.34), (0.38, PALETTE[0], 0.42)):
        shell = grid.contour([value], scalars="gyroid")
        plotter.add_mesh(shell, color=color, opacity=opacity, smooth_shading=True, specular=0.55)
    plotter.add_mesh(
        pv.Box(bounds=(-4.15, 4.15, -4.15, 4.15, -4.15, 4.15)),
        style="wireframe",
        color="#7896ab",
        opacity=0.17,
    )
    plotter.camera_position = [(12.2, -11.6, 9.0), (0, 0, 0), (0, 0, 1)]
    return plotter


def finite_element_stress() -> pv.Plotter:
    plotter = new_plotter("CANTILEVER STRESS", "FINITE-ELEMENT DISPLACEMENT / VON MISES ENVELOPE")
    x = np.linspace(0, 10, 88)
    y = np.linspace(-1.4, 1.4, 28)
    z = np.linspace(-0.55, 0.55, 16)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    displacement = -0.025 * xx**2 * (1 - 0.1 * yy**2)
    warped = pv.StructuredGrid(xx, yy, zz + displacement)
    stress = (1 - xx / 11) * (0.25 + np.abs(yy) / 1.4) + 0.12 * np.cos(xx * 2.4)
    warped["von_mises"] = stress.ravel(order="F")
    plotter.add_mesh(
        warped,
        scalars="von_mises",
        cmap=["#173f3e", "#54d6c6", "#ffd166", "#ff6f91"],
        smooth_shading=True,
        show_edges=True,
        edge_color="#d8f7ff",
        line_width=0.25,
        show_scalar_bar=False,
    )
    plotter.add_mesh(pv.Box(bounds=(-0.35, 0, -2.0, 2.0, -1.1, 1.1)), color="#7b9cff", opacity=0.32)
    for y0 in np.linspace(-1.1, 1.1, 5):
        plotter.add_mesh(pv.Arrow(start=(10, y0, -2.7), direction=(0, 0, -1), scale=0.9), color="#ff6f91")
    plotter.camera_position = [(13.5, -14.0, 8.5), (4.8, 0, -1.0), (0, 0, 1)]
    return plotter


def _bond(start: np.ndarray, end: np.ndarray, radius: float = 0.07) -> pv.PolyData:
    vector = end - start
    return pv.Cylinder(
        center=(start + end) / 2, direction=vector, radius=radius, height=float(np.linalg.norm(vector))
    )


def molecular_orbitals() -> pv.Plotter:
    plotter = new_plotter("MOLECULAR ORBITALS", "ATOM GEOMETRY / BONDS / ELECTRON-DENSITY LOBES")
    atoms = np.array(
        [
            (
                np.cos(i * 0.74) * (2.0 + 0.035 * i),
                np.sin(i * 0.74) * (1.7 + 0.025 * i),
                0.5 * np.sin(i * 1.13),
            )
            for i in range(24)
        ]
    )
    atom_colors = [PALETTE[i % len(PALETTE)] for i in range(len(atoms))]
    for i, point in enumerate(atoms):
        plotter.add_mesh(
            pv.Sphere(radius=0.18 + 0.035 * (i % 4), center=point),
            color=atom_colors[i],
            smooth_shading=True,
            specular=0.65,
        )
        if i:
            plotter.add_mesh(_bond(atoms[i - 1], point), color="#8fa9bc", smooth_shading=True)
    for i in range(3, 22, 6):
        plotter.add_mesh(_bond(atoms[i], atoms[(i + 7) % len(atoms)], 0.045), color="#ffd166")
    for center, direction, color in (
        ((-1.5, 0, 0), (1, 0.4, 0.1), PALETTE[3]),
        ((1.3, 0.3, 0), (-0.6, 1, 0.2), PALETTE[2]),
    ):
        for sign in (-1, 1):
            lobe = pv.ParametricEllipsoid(0.75, 0.35, 0.3)
            lobe.translate(np.asarray(center) + sign * np.asarray(direction) * 0.52, inplace=True)
            plotter.add_mesh(lobe, color=color, opacity=0.16, smooth_shading=True)
    plotter.camera_position = [(8.8, -9.8, 6.7), (0, 0, 0), (0, 0, 1)]
    return plotter


def classified_point_cloud() -> pv.Plotter:
    plotter = new_plotter("CLASSIFIED LIDAR", "GROUND / CANOPY / STRUCTURES / OUTLIERS")
    rng = np.random.default_rng(42)
    ground_xy = rng.uniform(-5, 5, size=(9000, 2))
    ground_z = 0.18 * np.sin(ground_xy[:, 0]) + 0.12 * np.cos(ground_xy[:, 1] * 1.7)
    ground = np.column_stack((ground_xy, ground_z))
    canopy_xy = rng.normal(size=(5200, 2)) * 2.8
    canopy_z = 1.2 + 2.8 * np.exp(-0.08 * np.sum(canopy_xy**2, axis=1)) + rng.normal(0, 0.35, len(canopy_xy))
    canopy = np.column_stack((canopy_xy, canopy_z))
    facade = np.column_stack((rng.uniform(-2.1, 2.1, 1800), np.full(1800, 3.4), rng.uniform(0.2, 4.5, 1800)))
    points = np.vstack((ground, canopy, facade))
    classes = np.concatenate((np.zeros(len(ground)), np.ones(len(canopy)), np.full(len(facade), 2)))
    cloud = pv.PolyData(points)
    cloud["class"] = classes
    plotter.add_mesh(
        cloud,
        scalars="class",
        cmap=[PALETTE[3], PALETTE[0], PALETTE[2]],
        clim=(0, 2),
        point_size=5,
        render_points_as_spheres=True,
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        pv.Box(bounds=(-2.3, 2.3, 3.2, 3.6, 0, 4.7)), style="wireframe", color="#ffd166", opacity=0.45
    )
    plotter.camera_position = [(10.5, -12.8, 8.5), (0, 0, 1.2), (0, 0, 1)]
    return plotter


def planetary_routes() -> pv.Plotter:
    plotter = new_plotter("PLANETARY ROUTES", "GEODESIC ARCS / STATION COVERAGE / ORBITAL SHELL")
    globe = pv.Sphere(radius=3.25, theta_resolution=96, phi_resolution=64)
    plotter.add_mesh(globe, color="#163252", opacity=0.62, smooth_shading=True, specular=0.5)
    plotter.add_mesh(globe, style="wireframe", color="#54d6c6", opacity=0.09, line_width=0.6)
    stations = np.array(
        [(2.6, 1.4, 1.3), (-2.4, 1.8, 1.2), (-1.6, -2.2, 1.7), (2.1, -2.2, -1.0), (0.3, 2.4, -2.1)]
    )
    stations = stations / np.linalg.norm(stations, axis=1)[:, None] * 3.28
    for i, start in enumerate(stations):
        plotter.add_mesh(pv.Sphere(radius=0.13, center=start), color=PALETTE[i], emissive=True)
        end = stations[(i + 2) % len(stations)]
        t = np.linspace(0, 1, 80)
        curve = (1 - t[:, None]) * start + t[:, None] * end
        curve = curve / np.linalg.norm(curve, axis=1)[:, None] * (3.3 + 1.1 * np.sin(np.pi * t))[:, None]
        plotter.add_mesh(pv.Spline(curve, 240).tube(radius=0.035), color=PALETTE[i], opacity=0.84)
    plotter.add_mesh(
        pv.Sphere(radius=4.65, theta_resolution=72, phi_resolution=48),
        style="wireframe",
        color="#7b9cff",
        opacity=0.06,
    )
    plotter.camera_position = [(9.8, -10.7, 7.4), (0, 0, 0), (0, 0, 1)]
    return plotter


def tensor_glyphs() -> pv.Plotter:
    plotter = new_plotter("TENSOR ORIENTATION FIELD", "PRINCIPAL DIRECTION / ANISOTROPY / MAGNITUDE")
    xx, yy, zz = np.meshgrid(
        np.linspace(-4, 4, 13), np.linspace(-3, 3, 10), np.linspace(-2, 2, 7), indexing="ij"
    )
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    cloud = pv.PolyData(points)
    vectors = np.column_stack((-points[:, 1], points[:, 0] + 0.4 * points[:, 2], 0.7 * np.sin(points[:, 0])))
    magnitude = np.linalg.norm(vectors, axis=1)
    vectors /= np.maximum(magnitude[:, None], 1e-6)
    cloud["vectors"] = vectors
    cloud["magnitude"] = magnitude
    glyphs = cloud.glyph(
        orient="vectors",
        scale="magnitude",
        factor=0.16,
        geom=pv.Arrow(tip_length=0.28, tip_radius=0.1, shaft_radius=0.035),
    )
    plotter.add_mesh(
        glyphs,
        scalars="magnitude",
        cmap=[PALETTE[0], PALETTE[3], PALETTE[2], PALETTE[1]],
        opacity=0.82,
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        pv.Box(bounds=(-4.4, 4.4, -3.4, 3.4, -2.4, 2.4)), style="wireframe", color="#7896ab", opacity=0.16
    )
    plotter.camera_position = [(12.5, -12.8, 9.5), (0, 0, 0), (0, 0, 1)]
    return plotter


def vascular_tree() -> pv.Plotter:
    plotter = new_plotter("VASCULAR TREE", "MULTISCALE BRANCHING / RADIUS TAPER / PERFUSION TERRITORY")
    rng = np.random.default_rng(7)
    frontier = [(np.array([0.0, 0.0, -3.5]), np.array([0.0, 0.0, 1.0]), 0.25)]
    for depth in range(6):
        next_frontier = []
        for origin, direction, radius in frontier:
            length = 1.25 - depth * 0.1
            end = origin + direction * length
            plotter.add_mesh(
                _bond(origin, end, radius), color=PALETTE[min(depth // 2, 4)], smooth_shading=True
            )
            for sign in (-1, 1):
                bend = np.array([sign * (0.42 + 0.08 * rng.random()), 0.22 * rng.normal(), 0.75])
                bend = bend / np.linalg.norm(bend)
                next_frontier.append((end, bend, radius * 0.69))
        frontier = next_frontier
    terminals = pv.PolyData(np.array([origin for origin, _, _ in frontier]))
    plotter.add_mesh(terminals, color="#ff6f91", point_size=9, render_points_as_spheres=True)
    plotter.add_mesh(pv.Sphere(radius=4.4, center=(0, 0, 0.2)), color="#54d6c6", opacity=0.035)
    plotter.camera_position = [(9.2, -11.2, 6.8), (0, 0, 0.2), (0, 0, 1)]
    return plotter


def seismic_slices() -> pv.Plotter:
    plotter = new_plotter("SEISMIC VOLUME", "ORTHOGONAL SLICES / FAULT PLANE / AMPLITUDE ENVELOPE")
    grid = pv.ImageData(dimensions=(82, 72, 64), spacing=(0.11, 0.12, 0.13), origin=(-4.45, -4.25, -4.1))
    x, y, z = grid.points.T
    layers = np.sin(5.2 * (z + 0.13 * x)) + 0.45 * np.sin(2.4 * y - 1.8 * z)
    fault = np.where(x + 0.4 * y > 0.6, 0.9, -0.4)
    amplitude = layers + fault + 1.6 * np.exp(-0.7 * ((x + 1.2) ** 2 + (y - 0.7) ** 2 + (z + 0.4) ** 2))
    grid["amplitude"] = amplitude
    slices = grid.slice_orthogonal(x=0.2, y=-0.35, z=0.45)
    plotter.add_mesh(
        slices,
        scalars="amplitude",
        cmap=["#172d54", "#54d6c6", "#eef8ff", "#ffd166", "#ff6f91"],
        opacity=0.88,
        show_scalar_bar=False,
    )
    fault_plane = pv.Plane(center=(0.6, 0, 0), direction=(1, 0.4, 0), i_size=8, j_size=8)
    plotter.add_mesh(fault_plane, color="#ff6f91", opacity=0.16)
    plotter.add_mesh(
        pv.Box(bounds=(-4.5, 4.5, -4.3, 4.3, -4.2, 4.2)), style="wireframe", color="#7896ab", opacity=0.18
    )
    plotter.camera_position = [(12.0, -13.0, 9.3), (0, 0, 0), (0, 0, 1)]
    return plotter


def urban_airflow() -> pv.Plotter:
    plotter = new_plotter("URBAN AIRFLOW", "BUILDING CANOPY / WIND CORRIDORS / TURBULENCE WAKE")
    rng = np.random.default_rng(19)
    for ix in range(-4, 5):
        for iy in range(-3, 4):
            if (ix + iy) % 4 == 0:
                continue
            height = 0.7 + 2.8 * rng.random()
            building = pv.Box(bounds=(ix - 0.33, ix + 0.33, iy - 0.33, iy + 0.33, 0, height))
            plotter.add_mesh(
                building,
                color=PALETTE[(ix + 2 * iy) % len(PALETTE)],
                opacity=0.38,
                show_edges=True,
                edge_color="#d8f7ff",
                line_width=0.3,
            )
    for lane in np.linspace(-2.8, 2.8, 11):
        x = np.linspace(-5.2, 5.2, 130)
        y = lane + 0.18 * np.sin(x * 1.2 + lane)
        z = 0.7 + 0.45 * np.sin(x * 0.55 + lane) ** 2
        path = pv.Spline(np.column_stack((x, y, z)), 320).tube(radius=0.025)
        plotter.add_mesh(path, color=PALETTE[int((lane + 3) * 2) % len(PALETTE)], opacity=0.72)
    plotter.add_mesh(
        pv.Plane(center=(0, 0, -0.02), direction=(0, 0, 1), i_size=11, j_size=8),
        color="#173f3e",
        opacity=0.22,
    )
    plotter.camera_position = [(11.8, -13.5, 9.2), (0, 0, 1.0), (0, 0, 1)]
    return plotter


BUILDERS = {
    "spectral-terrain": spectral_terrain,
    "vortex-streamlines": vortex_streamlines,
    "wave-isosurface": wave_isosurface,
    "gyroid-lattice": gyroid_lattice,
    "finite-element-stress": finite_element_stress,
    "molecular-orbitals": molecular_orbitals,
    "classified-point-cloud": classified_point_cloud,
    "planetary-routes": planetary_routes,
    "tensor-glyphs": tensor_glyphs,
    "vascular-tree": vascular_tree,
    "seismic-slices": seismic_slices,
    "urban-airflow": urban_airflow,
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
