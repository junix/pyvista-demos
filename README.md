# PyVista Demos

Scientific 3D scenes rendered off-screen through PyVista/VTK, with transparent PNG as the primary artifact.

| Scene | Preview | Data primitive |
|---|---|---|
| Spectral terrain | ![terrain](out/spectral-terrain-transparent.png) | Structured grid + contours |
| Vortex chamber | ![vortex](out/vortex-streamlines-transparent.png) | Vector field + stream tubes |
| Wave isosurfaces | ![wave](out/wave-isosurface-transparent.png) | Volumetric scalar field + contours |

```bash
uv sync
uv run render-pyvista-demos
uv run ruff check .
uv run pytest
```

The renderer uses fixed cameras and deterministic fields, requests VTK's transparent screenshot path, and validates alpha plus non-trivial chromatic content.
