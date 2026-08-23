# PyVista Demos

Twelve scientific 3D reference scenes rendered off-screen through PyVista/VTK, with transparent PNG as the primary artifact.

`catalog.json` records the scientific use, question, visual family, complexity, and data/renderer tags.

| Terrain | Vortex | Waves | Gyroid |
|---|---|---|---|
| ![terrain](out/spectral-terrain-transparent.png) | ![vortex](out/vortex-streamlines-transparent.png) | ![wave](out/wave-isosurface-transparent.png) | ![gyroid](out/gyroid-lattice-transparent.png) |
| FEA stress | Molecular orbitals | Point cloud | Planetary routes |
| ![fea](out/finite-element-stress-transparent.png) | ![molecule](out/molecular-orbitals-transparent.png) | ![point cloud](out/classified-point-cloud-transparent.png) | ![routes](out/planetary-routes-transparent.png) |
| Tensor glyphs | Vascular tree | Seismic slices | Urban airflow |
| ![tensors](out/tensor-glyphs-transparent.png) | ![vascular](out/vascular-tree-transparent.png) | ![seismic](out/seismic-slices-transparent.png) | ![airflow](out/urban-airflow-transparent.png) |

```bash
uv sync
uv run render-pyvista-demos
uv run ruff check .
uv run pytest
```

The renderer uses fixed cameras and deterministic fields, requests VTK's transparent screenshot path, and validates alpha plus non-trivial chromatic content.
