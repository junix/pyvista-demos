from pathlib import Path

from pyvista_demos import SCENES
from pyvista_demos.render import BUILDERS, validate_png


def test_every_scene_has_a_builder() -> None:
    assert len(SCENES) >= 12
    assert set(SCENES) == set(BUILDERS)


def test_rendered_gallery_contract() -> None:
    outputs = [Path("out") / f"{scene}-transparent.png" for scene in SCENES]
    assert all(path.exists() for path in outputs)
    assert all(validate_png(path)["colorful"] > 2500 for path in outputs)
