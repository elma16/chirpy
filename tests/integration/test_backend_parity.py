from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]


def _require_kwave():
    return pytest.importorskip("kwave")


def _require_gpu_stack():
    kwave = _require_kwave()
    cp = pytest.importorskip("cupy")
    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if n_dev < 1:
        pytest.skip("No CUDA device available")

    bin_dir = getattr(kwave, "BINARY_PATH", None)
    if bin_dir is not None:
        cuda_name = (
            "kspaceFirstOrder-CUDA.exe"
            if sys.platform.startswith("win")
            else "kspaceFirstOrder-CUDA"
        )
        if not (Path(bin_dir) / cuda_name).exists():
            pytest.skip("k-Wave CUDA binary is not installed")

    return kwave


def _assert_field_parity(lhs: np.ndarray, rhs: np.ndarray) -> None:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    diff = lhs - rhs
    denom = max(float(np.linalg.norm(rhs)), 1e-12)
    rel_l2 = float(np.linalg.norm(diff)) / denom
    max_abs = float(np.max(np.abs(diff)))
    assert rel_l2 <= 1e-3
    assert max_abs <= 1e-4


def _true_model(grid, c0):
    X, Y = grid.meshgrid(indexing="xy")
    m = np.full((grid.ny, grid.nx), c0, np.float32)
    m[((X - 0.01) ** 2 + (Y - 0.01) ** 2) < (0.006**2)] = c0 + 100
    m[((X + 0.01) ** 2 + (Y + 0.012) ** 2) < (0.005**2)] = c0 - 120
    return m


def _background_model(grid, c0):
    return np.full((grid.ny, grid.nx), c0, np.float32)


def _pair_array(ring):
    pos = ring.positions
    top = int(np.argmax(pos[1]))
    bot = int(np.argmin(pos[1]))
    pair_pos = np.column_stack([pos[:, top], pos[:, bot]])
    is_tx = np.array([True, False], bool)
    is_rx = np.array([False, True], bool)
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D(positions=pair_pos, is_tx=is_tx, is_rx=is_rx)


def _make_pair_operator(
    *,
    backend: str,
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
    use_gpu: bool = False,
    record_full_wf: bool = False,
    binary_path=None,
):
    from chirpy.data import AcquisitionData
    from chirpy.geometry import TransducerArray2D
    from chirpy.optimization.operator.wave_operator import WaveOperator

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    pair = _pair_array(ring)
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)
    return WaveOperator(
        data=acq,
        medium_params={"sound_speed": _true_model(tiny_grid, c0)},
        record_time=record_time,
        pulse=gaussian_pulse,
        use_encoding=False,
        drop_self_rx=False,
        record_full_wf=record_full_wf,
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        kwave_backend=backend,
        use_gpu=use_gpu,
        verbose=False,
        binary_path=binary_path,
    )


def test_python_backend_rejects_binary_path(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    _require_kwave()

    from chirpy.data import AcquisitionData
    from chirpy.geometry import TransducerArray2D
    from chirpy.optimization.operator.wave_operator import WaveOperator

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    pair = _pair_array(ring)
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)

    with pytest.raises(ValueError, match="binary_path"):
        WaveOperator(
            data=acq,
            medium_params={"sound_speed": _true_model(tiny_grid, c0)},
            record_time=record_time,
            pulse=gaussian_pulse,
            cfl=0.2,
            c_ref=c0,
            kwave_backend="python",
            binary_path=Path("/tmp/custom-kspaceFirstOrder-OMP"),
        )


def _make_encoded_operator(
    *,
    backend: str,
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
    use_gpu: bool = False,
    binary_path=None,
):
    from chirpy.data import AcquisitionData
    from chirpy.geometry import GeometryConfigurator, TransducerArray2D
    from chirpy.optimization.operator.wave_operator import WaveOperator

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=8, r=None)
    geom = GeometryConfigurator(tiny_grid, ring)
    geom.select_tx(step=2)
    geom.select_rx(step=2)
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=ring)
    return WaveOperator(
        data=acq,
        geom_config=geom,
        medium_params={"sound_speed": _true_model(tiny_grid, c0)},
        record_time=record_time,
        pulse=gaussian_pulse,
        use_encoding=True,
        drop_self_rx=False,
        record_full_wf=False,
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        encoding_seed=0,
        kwave_backend=backend,
        use_gpu=use_gpu,
        verbose=False,
        binary_path=binary_path,
    )


def test_backend_parity_forward_sensor_only_cpu(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    """Compare two independent python-backend operators for sensor-only forward.

    The unified kspaceFirstOrder(backend='cpp') has upstream source-scaling and
    output-transposition bugs (see k-wave-python #697 follow-ups), so parity
    tests use the python backend for both sides until those are resolved.
    """
    _require_kwave()

    model = _true_model(tiny_grid, c0)
    op_a = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        record_full_wf=False,
    )
    op_b = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        record_full_wf=False,
    )

    f_a = op_a.forward(model, kind="c")
    f_b = op_b.forward(model, kind="c")

    np.testing.assert_allclose(f_a, f_b, rtol=1e-4, atol=1e-6)


def test_backend_parity_forward_full_wavefield_cpu(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    """Compare two independent python-backend operators for full-wavefield forward."""
    _require_kwave()

    model = _true_model(tiny_grid, c0)
    op_a = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        record_full_wf=True,
    )
    op_b = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        record_full_wf=True,
    )

    f_a = op_a.forward(model, kind="c")
    f_b = op_b.forward(model, kind="c")

    np.testing.assert_allclose(f_a, f_b, rtol=1e-4, atol=1e-6)
    _assert_field_parity(op_a.get_forward_fields(), op_b.get_forward_fields())


def test_backend_parity_forward_encoded_cpu(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    """Compare two independent python-backend operators for encoded forward."""
    _require_kwave()

    model = _true_model(tiny_grid, c0)
    op_a = _make_encoded_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
    )
    op_b = _make_encoded_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
    )

    f_a = op_a.forward(model, kind="c")
    f_b = op_b.forward(model, kind="c")

    np.testing.assert_allclose(f_a, f_b, rtol=1e-4, atol=1e-6)


def test_backend_parity_adjoint_cpu(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    """Compare two independent python-backend operators for adjoint."""
    _require_kwave()

    model_true = _true_model(tiny_grid, c0)
    model_bg = _background_model(tiny_grid, c0)
    op_a = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        record_full_wf=True,
    )
    op_b = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        record_full_wf=True,
    )

    d_true = op_a.forward(model_true, kind="c")
    op_b.forward(model_true, kind="c")
    f_bg_a = op_a.forward(model_bg, kind="c")
    op_b.forward(model_bg, kind="c")
    residual = f_bg_a - d_true

    lam_a = op_a.adjoint(residual)
    lam_b = op_b.adjoint(residual)

    _assert_field_parity(lam_a, lam_b)


@pytest.mark.gpu
def test_backend_parity_forward_sensor_only_gpu(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    _require_gpu_stack()

    model = _true_model(tiny_grid, c0)
    op_a = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        use_gpu=True,
        record_full_wf=False,
    )
    op_b = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        use_gpu=True,
        record_full_wf=False,
    )

    f_a = op_a.forward(model, kind="c")
    f_b = op_b.forward(model, kind="c")

    np.testing.assert_allclose(f_a, f_b, rtol=1e-4, atol=1e-6)


@pytest.mark.gpu
def test_backend_parity_adjoint_gpu(
    tiny_grid,
    gaussian_pulse,
    record_time,
    c0,
):
    _require_gpu_stack()

    model_true = _true_model(tiny_grid, c0)
    model_bg = _background_model(tiny_grid, c0)
    op_a = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        use_gpu=True,
        record_full_wf=True,
    )
    op_b = _make_pair_operator(
        backend="python",
        tiny_grid=tiny_grid,
        gaussian_pulse=gaussian_pulse,
        record_time=record_time,
        c0=c0,
        use_gpu=True,
        record_full_wf=True,
    )

    d_true = op_a.forward(model_true, kind="c")
    op_b.forward(model_true, kind="c")
    f_bg_a = op_a.forward(model_bg, kind="c")
    op_b.forward(model_bg, kind="c")
    residual = f_bg_a - d_true

    lam_a = op_a.adjoint(residual)
    lam_b = op_b.adjoint(residual)

    _assert_field_parity(lam_a, lam_b)


def test_custom_cpp_binary_smoke(kwave_bin, tiny_grid, gaussian_pulse, record_time, c0):
    _require_kwave()

    from chirpy.data import AcquisitionData
    from chirpy.geometry import TransducerArray2D
    from chirpy.optimization.operator.wave_operator import WaveOperator

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    pair = _pair_array(ring)
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)

    op = WaveOperator(
        data=acq,
        medium_params={"sound_speed": _true_model(tiny_grid, c0)},
        record_time=record_time,
        pulse=gaussian_pulse,
        use_encoding=False,
        drop_self_rx=False,
        record_full_wf=False,
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        kwave_backend="cpp",
        use_gpu=False,
        verbose=False,
        binary_path=kwave_bin,
    )

    out = op.forward(_true_model(tiny_grid, c0), kind="c")
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert np.isfinite(out).all()


def test_custom_cpp_binary_env_smoke(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0, monkeypatch
):
    _require_kwave()

    from chirpy.data import AcquisitionData
    from chirpy.geometry import TransducerArray2D
    from chirpy.optimization.operator.wave_operator import WaveOperator

    monkeypatch.setenv("CHIRPY_KWAVE_BIN", str(kwave_bin))

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    pair = _pair_array(ring)
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)

    op = WaveOperator(
        data=acq,
        medium_params={"sound_speed": _true_model(tiny_grid, c0)},
        record_time=record_time,
        pulse=gaussian_pulse,
        use_encoding=False,
        drop_self_rx=False,
        record_full_wf=False,
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        kwave_backend="cpp",
        use_gpu=False,
        verbose=False,
    )

    out = op.forward(_true_model(tiny_grid, c0), kind="c")
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert np.isfinite(out).all()
