#!/usr/bin/env python3
import numpy as np

from chirpy.utils.visualizer_multi_mode import Visualizer


def _tiny_fields():
    nx, ny = 8, 6
    xi = np.linspace(-0.01, 0.01, nx)
    yi = np.linspace(-0.01, 0.01, ny)

    X, Y = np.meshgrid(xi, yi, indexing="xy")
    C_true = 1500.0 + 10.0 * np.sin(X * 100.0) * np.cos(Y * 80.0)
    atten_true = 0.5 + 0.1 * np.sin(X * 50.0)

    return xi, yi, C_true.astype(np.float32), atten_true.astype(np.float32)


def test_visualizer_init_and_single_update_both_mode():
    xi, yi, C_true, atten_true = _tiny_fields()

    vis = Visualizer(
        xi=xi,
        yi=yi,
        C_true=C_true,
        atten_true=atten_true,
        mode="both",
        baseline=1500.0,
        sync_clim=True,
    )

    # First update with all fields present
    vel_est = C_true + 5.0
    atten_est = atten_true + 0.05
    grad = np.ones_like(C_true) * 0.1
    search = -grad

    vis.update(
        vel_est=vel_est,
        atten_est=atten_est,
        grad=grad,
        search_dir=search,
        title="unit-test",
    )

    assert vis.global_iter == 1
    # Check some of the image handles exist
    assert vis.im_est_vel is not None
    assert vis.im_grad_vel is not None
    assert vis.im_est_atten is not None
    assert vis.im_search is not None


def test_visualizer_vel_only_mode_and_safe_clim():
    xi, yi, C_true, atten_true = _tiny_fields()

    vis = Visualizer(
        xi=xi,
        yi=yi,
        C_true=C_true,
        atten_true=atten_true,
        mode="vel",
        sync_clim=False,
    )

    # Constant image → _safe_clim should pad range
    img = np.ones_like(C_true)
    vmin, vmax = vis._safe_clim(img)
    assert vmin < vmax

    # Only velocity / gradient / search are relevant in vel mode
    vis.update(
        vel_est=C_true,
        grad=C_true * 0.0,
        search_dir=C_true * 0.0,
    )
    assert vis.global_iter == 1


def test_visualizer_invalid_mode_raises():
    xi, yi, C_true, atten_true = _tiny_fields()
    try:
        Visualizer(
            xi=xi,
            yi=yi,
            C_true=C_true,
            atten_true=atten_true,
            mode="not-a-mode",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid mode")
