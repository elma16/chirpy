#!/usr/bin/env python3
import numpy as np

from chirpy.data import ImageData
from chirpy.geometry import ImageGrid2D


def _make_grid_and_image(nx=16, ny=12):
    grid = ImageGrid2D(nx=nx, ny=ny, dx=1.0)
    # Simple smooth pattern so interpolation is well-behaved
    X, Y = grid.meshgrid(indexing="xy")
    arr = np.sin(X / 5.0) * np.cos(Y / 7.0)
    return grid, arr


def test_image_data_history_and_max_history():
    grid, arr = _make_grid_and_image()
    img = ImageData(arr, grid=grid, max_history=2)

    assert len(img.history) == 1
    np.testing.assert_allclose(img.current, arr)

    img.update(arr * 2.0)
    assert len(img.history) == 2
    np.testing.assert_allclose(img.history[-1], arr * 2.0)

    # Exceed max_history → oldest snapshot dropped
    img.update(arr * 3.0)
    assert len(img.history) == 2
    np.testing.assert_allclose(img.history[-1], arr * 3.0)


def test_image_data_downsample_and_upsample_roundtrip():
    grid, arr = _make_grid_and_image(nx=20, ny=20)
    img = ImageData(arr, grid=grid)

    # Coarser grid (half resolution)
    coarse = ImageGrid2D(nx=10, ny=10, dx=grid.dx * 2.0)
    # The implementation enforces matching extents unless you explicitly allow
    # stretching, so we opt in here to exercise the zoom-based downsample path.
    down = img.downsample_to(coarse, allow_stretch=True)
    assert down.array.shape == (coarse.ny, coarse.nx)

    # Finer grid (double resolution)
    fine = ImageGrid2D(nx=40, ny=40, dx=grid.dx * 0.5)
    up = img.upsample_to(fine)
    assert up.array.shape == (fine.ny, fine.nx)

    # Both down and up should be finite everywhere
    assert np.isfinite(down.array).all()
    assert np.isfinite(up.array).all()


def test_image_data_resample_interp_vs_zoom_modes():
    grid, arr = _make_grid_and_image(nx=8, ny=8)
    img = ImageData(arr, grid=grid)

    finer = ImageGrid2D(nx=16, ny=16, dx=grid.dx / 2.0)

    # interp mode: uses RegularGridInterpolator
    up_interp = img._resample_to(finer, mode="interp", method="linear")
    assert up_interp.array.shape == (finer.ny, finer.nx)
    assert up_interp.ctx["resample"]["mode"].startswith("interp")

    # zoom mode: pixel scaling (must allow stretching if extents differ)
    up_zoom = img._resample_to(
        finer, mode="zoom", order=1, allow_stretch=True, copy_ctx=False
    )
    assert up_zoom.array.shape == (finer.ny, finer.nx)
    assert up_zoom.ctx["resample"]["mode"] == "zoom"
