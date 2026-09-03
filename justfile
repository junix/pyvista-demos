set shell := ["bash", "-euo", "pipefail", "-c"]

install_bin := home_directory() / "sync" / "bin"

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

# Interpreted CLI launcher (ADR-749).
install: build
    #!/usr/bin/env bash
    mkdir -p "{{ install_bin }}"
    cat > "{{ install_bin }}/render-pyvista-demos" << 'EOF'
    #!/usr/bin/env bash
    cd {{ justfile_directory() }} && uv run render-pyvista-demos "$@"
    EOF
    chmod +x "{{ install_bin }}/render-pyvista-demos"

# Remove generated images.
clean:
    rm -rf out
    mkdir -p out
    touch out/.gitkeep

# Rebuild gallery.html from catalog.json, README.md and the artifacts in out/.
gallery:
    python3 tools/gallery.py
