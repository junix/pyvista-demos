import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyvista_demos import SCENES
from pyvista_demos import render as render_module
from pyvista_demos.render import (
    BUILDERS,
    _bond,
    main,
    new_plotter,
    validate_png,
)

TRANSPARENT = (0, 0, 0, 0)
COLORFUL = (10, 200, 60, 255)
GRAY = (100, 100, 100, 255)
FAINT = (10, 200, 60, 15)


def _write_png(
    path: Path, bands: list[tuple[int, tuple[int, int, int, int]]], width: int = 100, height: int = 100
) -> None:
    assert sum(count for count, _ in bands) == width * height
    image = Image.new("RGBA", (width, height))
    pixels = image.load()
    index = 0
    for count, rgba in bands:
        for _ in range(count):
            pixels[index % width, index // width] = rgba
            index += 1
    image.save(path)


def _spy_render(calls: list[tuple[str, Path]]):
    def _render(scene: str, output: Path) -> dict[str, object]:
        calls.append((scene, output))
        return {"scene": scene, "output": str(output)}

    return _render


def test_every_scene_has_a_builder() -> None:
    assert len(SCENES) >= 12
    assert set(SCENES) == set(BUILDERS)


def test_rendered_gallery_contract() -> None:
    outputs = [Path("out") / f"{scene}-transparent.png" for scene in SCENES]
    assert all(path.exists() for path in outputs)
    assert all(validate_png(path)["colorful"] > 2500 for path in outputs)
    for scene, path in zip(SCENES, outputs):
        report = validate_png(path)
        assert report["scene"] == scene
        assert report["size"] == (1600, 1000)
        assert 6.0 <= report["transparent_pct"] <= 100.0
        assert 3.5 <= report["visible_pct"] <= 100.0


def test_validate_png_reports_exact_metrics(tmp_path: Path) -> None:
    path = tmp_path / "demo-transparent.png"
    _write_png(path, [(1000, TRANSPARENT), (9000, COLORFUL)])
    assert validate_png(path) == {
        "scene": "demo",
        "size": (100, 100),
        "transparent_pct": 10.0,
        "visible_pct": 90.0,
        "colorful": 9000,
    }


@pytest.mark.parametrize(
    ("filename", "scene"),
    [
        ("demo-transparent.png", "demo"),
        ("plain.png", "plain"),
        ("grid-transparent-transparent.png", "grid-transparent"),
    ],
)
def test_validate_png_derives_scene_name(tmp_path: Path, filename: str, scene: str) -> None:
    path = tmp_path / filename
    _write_png(path, [(1000, TRANSPARENT), (9000, COLORFUL)])
    assert validate_png(path)["scene"] == scene


@pytest.mark.parametrize(
    ("name", "bands", "message"),
    [
        ("fully opaque", [(10000, COLORFUL)], "t=0 v=10000 c=10000"),
        ("near-invisible", [(1000, TRANSPARENT), (9000, (10, 200, 60, 8))], "t=1000 v=0 c=0"),
        ("monochrome", [(1000, TRANSPARENT), (9000, GRAY)], "t=1000 v=9000 c=0"),
        (
            "alpha 16 is visible but not colorful",
            [(1000, TRANSPARENT), (9000, (10, 200, 60, 16))],
            "t=1000 v=9000 c=0",
        ),
    ],
)
def test_validate_png_rejects_weak_content(
    tmp_path: Path, name: str, bands: list[tuple[int, tuple[int, int, int, int]]], message: str
) -> None:
    path = tmp_path / "weak-transparent.png"
    _write_png(path, bands)
    with pytest.raises(RuntimeError) as excinfo:
        validate_png(path)
    assert excinfo.value.args[0] == f"weak-transparent.png: weak RGBA content {message}"


@pytest.mark.parametrize(
    ("name", "bands", "expected"),
    [
        (
            "transparent fraction exactly 6% passes",
            [(600, TRANSPARENT), (9400, COLORFUL)],
            {
                "scene": "boundary",
                "size": (100, 100),
                "transparent_pct": 6.0,
                "visible_pct": 94.0,
                "colorful": 9400,
            },
        ),
        (
            "one pixel below 6% transparency raises",
            [(599, TRANSPARENT), (9401, COLORFUL)],
            "boundary-transparent.png: weak RGBA content t=599 v=9401 c=9401",
        ),
        (
            "colorful count exactly 2500 passes",
            [(1000, TRANSPARENT), (2500, COLORFUL), (6500, GRAY)],
            {
                "scene": "boundary",
                "size": (100, 100),
                "transparent_pct": 10.0,
                "visible_pct": 90.0,
                "colorful": 2500,
            },
        ),
        (
            "one pixel below 2500 colorful raises",
            [(1000, TRANSPARENT), (2499, COLORFUL), (6501, GRAY)],
            "boundary-transparent.png: weak RGBA content t=1000 v=9000 c=2499",
        ),
        (
            "channel spread exactly 24 is not colorful",
            [(1000, TRANSPARENT), (9000, (0, 24, 0, 255))],
            "boundary-transparent.png: weak RGBA content t=1000 v=9000 c=0",
        ),
        (
            "channel spread 25 is colorful",
            [(1000, TRANSPARENT), (9000, (0, 25, 0, 255))],
            {
                "scene": "boundary",
                "size": (100, 100),
                "transparent_pct": 10.0,
                "visible_pct": 90.0,
                "colorful": 9000,
            },
        ),
        (
            "alpha exactly 17 counts as colorful",
            [(1000, TRANSPARENT), (9000, (10, 200, 60, 17))],
            {
                "scene": "boundary",
                "size": (100, 100),
                "transparent_pct": 10.0,
                "visible_pct": 90.0,
                "colorful": 9000,
            },
        ),
        (
            "alpha exactly 15 is not visible",
            [(1000, TRANSPARENT), (100, (10, 200, 60, 15)), (8900, COLORFUL)],
            {
                "scene": "boundary",
                "size": (100, 100),
                "transparent_pct": 10.0,
                "visible_pct": 89.0,
                "colorful": 8900,
            },
        ),
    ],
)
def test_validate_png_threshold_boundaries(
    tmp_path: Path,
    name: str,
    bands: list[tuple[int, tuple[int, int, int, int]]],
    expected: dict[str, object] | str,
) -> None:
    path = tmp_path / "boundary-transparent.png"
    _write_png(path, bands)
    if isinstance(expected, str):
        with pytest.raises(RuntimeError) as excinfo:
            validate_png(path)
        assert excinfo.value.args[0] == expected
    else:
        assert validate_png(path) == expected


@pytest.mark.parametrize(
    ("name", "bands", "expected"),
    [
        (
            "all three metrics exactly at their floor passes",
            [(7200, TRANSPARENT), (108600, FAINT), (1700, GRAY), (2500, COLORFUL)],
            {
                "scene": "limit",
                "size": (400, 300),
                "transparent_pct": 6.0,
                "visible_pct": 3.5,
                "colorful": 2500,
            },
        ),
        (
            "one visible pixel below 3.5% raises while the other two metrics pass",
            [(7200, TRANSPARENT), (108601, FAINT), (1699, GRAY), (2500, COLORFUL)],
            "limit-transparent.png: weak RGBA content t=7200 v=4199 c=2500",
        ),
    ],
)
def test_validate_png_visible_floor_boundary(
    tmp_path: Path,
    name: str,
    bands: list[tuple[int, tuple[int, int, int, int]]],
    expected: dict[str, object] | str,
) -> None:
    path = tmp_path / "limit-transparent.png"
    _write_png(path, bands, width=400, height=300)
    if isinstance(expected, str):
        with pytest.raises(RuntimeError) as excinfo:
            validate_png(path)
        assert excinfo.value.args[0] == expected
    else:
        assert validate_png(path) == expected


def test_validate_png_reports_size_as_width_height(tmp_path: Path) -> None:
    path = tmp_path / "rect-transparent.png"
    _write_png(path, [(600, TRANSPARENT), (5400, COLORFUL)], width=120, height=50)
    assert validate_png(path) == {
        "scene": "rect",
        "size": (120, 50),
        "transparent_pct": 10.0,
        "visible_pct": 90.0,
        "colorful": 5400,
    }


def test_validate_png_rounds_percentages_to_one_decimal(tmp_path: Path) -> None:
    path = tmp_path / "round-transparent.png"
    _write_png(path, [(3331, TRANSPARENT), (6669, COLORFUL)])
    assert validate_png(path) == {
        "scene": "round",
        "size": (100, 100),
        "transparent_pct": 33.3,
        "visible_pct": 66.7,
        "colorful": 6669,
    }


def test_new_plotter_applies_shared_scene_dressing() -> None:
    plotter = new_plotter("TITLE", "SUBTITLE")
    try:
        assert tuple(plotter.window_size) == (1600, 1000)
        assert plotter.off_screen is True
        assert plotter.background_color.hex_rgba == "#07111fff"
        assert len(plotter.actors) == 2
        title, subtitle = sorted(
            plotter.actors.values(), key=lambda actor: actor.position[1], reverse=True
        )
        assert title.GetInput() == "TITLE"
        assert tuple(title.position) == (55.0, 925.0)
        assert title.prop.color.hex_rgba == "#edf8ffff"
        assert subtitle.GetInput() == "SUBTITLE"
        assert tuple(subtitle.position) == (58.0, 886.0)
        assert subtitle.prop.color.hex_rgba == "#82a0b5ff"
    finally:
        plotter.close()


@pytest.mark.parametrize(
    ("start", "end", "radius", "expected_radius"),
    [
        ((1.0, 2.0, 3.0), (1.0, 2.0, 8.0), None, 0.07),
        ((1.0, 2.0, 3.0), (4.0, 6.0, 3.0), 0.05, 0.05),
    ],
)
def test_bond_spans_endpoints(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float | None,
    expected_radius: float,
) -> None:
    begin, finish = np.asarray(start), np.asarray(end)
    bond = _bond(begin, finish) if radius is None else _bond(begin, finish, radius=radius)
    axis = (finish - begin) / np.linalg.norm(finish - begin)
    projection = (bond.points - begin) @ axis
    assert projection.min() == pytest.approx(0.0, abs=1e-6)
    assert projection.max() == pytest.approx(float(np.linalg.norm(finish - begin)))
    assert bond.center == pytest.approx(tuple((begin + finish) / 2))
    offsets = np.linalg.norm(bond.points - (begin + np.outer(projection, axis)), axis=1)
    assert offsets.max() == pytest.approx(expected_radius, abs=1e-3)


def test_main_renders_every_scene_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(render_module, "render", _spy_render(calls))
    out_dir = tmp_path / "gallery"
    monkeypatch.setattr(sys, "argv", ["render-pyvista-demos", "--out", str(out_dir)])
    main()
    assert calls == [(scene, out_dir / f"{scene}-transparent.png") for scene in SCENES]
    assert out_dir.is_dir()
    expected = [
        {"scene": scene, "output": str(out_dir / f"{scene}-transparent.png")} for scene in SCENES
    ]
    out = capsys.readouterr().out
    assert out == json.dumps(expected, indent=2) + "\n"
    assert json.loads(out) == expected


def test_main_renders_only_the_selected_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(render_module, "render", _spy_render(calls))
    out_dir = tmp_path / "gallery"
    monkeypatch.setattr(
        sys, "argv", ["render-pyvista-demos", "--scene", "gyroid-lattice", "--out", str(out_dir)]
    )
    main()
    assert calls == [("gyroid-lattice", out_dir / "gyroid-lattice-transparent.png")]
    assert json.loads(capsys.readouterr().out) == [
        {"scene": "gyroid-lattice", "output": str(out_dir / "gyroid-lattice-transparent.png")}
    ]


def test_main_defaults_output_to_out_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(render_module, "render", _spy_render(calls))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["render-pyvista-demos"])
    main()
    assert calls == [(scene, Path("out") / f"{scene}-transparent.png") for scene in SCENES]
    assert (tmp_path / "out").is_dir()


def test_main_creates_nested_output_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(render_module, "render", _spy_render(calls))
    out_dir = tmp_path / "deep" / "nested" / "gallery"
    monkeypatch.setattr(
        sys, "argv", ["render-pyvista-demos", "--scene", "gyroid-lattice", "--out", str(out_dir)]
    )
    main()
    assert calls == [("gyroid-lattice", out_dir / "gyroid-lattice-transparent.png")]
    assert out_dir.is_dir()


def test_main_renders_into_an_existing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(render_module, "render", _spy_render(calls))
    out_dir = tmp_path / "gallery"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale")
    monkeypatch.setattr(
        sys, "argv", ["render-pyvista-demos", "--scene", "gyroid-lattice", "--out", str(out_dir)]
    )
    main()
    assert calls == [("gyroid-lattice", out_dir / "gyroid-lattice-transparent.png")]
    assert (out_dir / "stale.txt").read_text() == "stale"


def test_main_rejects_unknown_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(render_module, "render", _spy_render(calls))
    out_dir = tmp_path / "never-created"
    monkeypatch.setattr(
        sys, "argv", ["render-pyvista-demos", "--scene", "nope", "--out", str(out_dir)]
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert calls == []
    assert not out_dir.exists()


def test_main_aborts_gallery_when_a_scene_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path]] = []

    def _render(scene: str, output: Path) -> dict[str, object]:
        calls.append((scene, output))
        if scene == "wave-isosurface":
            raise RuntimeError("wave-isosurface: weak RGBA content t=0 v=0 c=0")
        return {"scene": scene, "output": str(output)}

    monkeypatch.setattr(render_module, "render", _render)
    out_dir = tmp_path / "gallery"
    monkeypatch.setattr(sys, "argv", ["render-pyvista-demos", "--out", str(out_dir)])
    with pytest.raises(RuntimeError) as excinfo:
        main()
    assert excinfo.value.args[0] == "wave-isosurface: weak RGBA content t=0 v=0 c=0"
    assert calls == [(scene, out_dir / f"{scene}-transparent.png") for scene in SCENES[:3]]
    assert capsys.readouterr().out == ""


def test_render_drives_the_plotter_lifecycle_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[object, ...]] = []
    builds: list[int] = []

    class _SpyPlotter:
        def show(self, auto_close: bool) -> None:
            events.append(("show", auto_close))

        def screenshot(self, output: Path, transparent_background: bool) -> None:
            events.append(("screenshot", output, transparent_background))
            _write_png(output, [(6000, TRANSPARENT), (54000, COLORFUL)], width=200, height=300)

        def close(self) -> None:
            events.append(("close",))

    def _builder() -> _SpyPlotter:
        builds.append(1)
        return _SpyPlotter()

    monkeypatch.setitem(BUILDERS, "stub-scene", _builder)
    output = tmp_path / "stub-scene-transparent.png"
    report = render_module.render("stub-scene", output)
    assert builds == [1]
    assert events == [
        ("show", False),
        ("screenshot", output, True),
        ("close",),
    ]
    assert report == {
        "scene": "stub-scene",
        "size": (200, 300),
        "transparent_pct": 10.0,
        "visible_pct": 90.0,
        "colorful": 54000,
    }


def test_render_rejects_weak_screenshot_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[object, ...]] = []

    class _WeakPlotter:
        def show(self, auto_close: bool) -> None:
            events.append(("show", auto_close))

        def screenshot(self, output: Path, transparent_background: bool) -> None:
            events.append(("screenshot", output, transparent_background))
            _write_png(output, [(60000, COLORFUL)], width=200, height=300)

        def close(self) -> None:
            events.append(("close",))

    monkeypatch.setitem(BUILDERS, "stub-scene", _WeakPlotter)
    output = tmp_path / "stub-scene-transparent.png"
    with pytest.raises(RuntimeError) as excinfo:
        render_module.render("stub-scene", output)
    assert excinfo.value.args[0] == "stub-scene-transparent.png: weak RGBA content t=0 v=60000 c=60000"
    assert events == [
        ("show", False),
        ("screenshot", output, True),
        ("close",),
    ]
    assert output.exists()


def test_render_rejects_unknown_scene(tmp_path: Path) -> None:
    output = tmp_path / "nope-transparent.png"
    with pytest.raises(KeyError) as excinfo:
        render_module.render("nope", output)
    assert excinfo.value.args[0] == "nope"
    assert not output.exists()
