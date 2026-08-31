set shell := ["bash", "-euo", "pipefail", "-c"]

default: build

# Sync deps and render every demo into out/ with fixed cameras and
# deterministic fields; screenshots are validated for alpha and chroma.
build:
    uv sync
    uv run render-pyvista-demos

# Render, then lint and run the test suite.
test: build
    uv run ruff check .
    uv run pytest

# Demos repo — no binary, no launcher (ADR-749: nothing to install).
install:
    @echo "pyvista-demos: demos repo, nothing to install"

# Remove generated images.
clean:
    rm -rf out
    mkdir -p out
    touch out/.gitkeep

# Rebuild gallery.html from catalog.json, README.md and the artifacts in out/.
gallery:
    python3 tools/gallery.py
