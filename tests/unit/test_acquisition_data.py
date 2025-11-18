#!/usr/bin/env python3
import numpy as np

from chirpy.data import AcquisitionData
from chirpy.geometry import ImageGrid2D, TransducerArray2D


def _make_small_acq(T=16):
    grid = ImageGrid2D(nx=8, ny=8, dx=1.0)
    # 2 elements, both TX/RX
    tarr = TransducerArray2D(
        positions=np.c_[[-0.5, 0.5], [0.0, 0.0]],
        is_tx=[True, True],
        is_rx=[True, True],
    )
    time = np.linspace(0.0, 1e-4, T, dtype=float)
    arr = np.random.randn(2, 2, T).astype(np.float32)
    acq = AcquisitionData(array=arr, grid=grid, tx_array=tarr, time=time, c0=1500.0)
    return acq


def test_set_time_and_freqs_validation():
    acq = _make_small_acq(T=16)

    # Valid re-assign
    t = np.linspace(0.0, 1e-4, 16)
    acq.set_time(t)
    assert acq.time.shape == (16,)
    assert acq.freqs is None

    # Mismatched length → ValueError
    t_bad = np.linspace(0.0, 1e-4, 15)
    try:
        acq.set_time(t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched time length")

    # Switch to frequency axis
    f = np.linspace(0.0, 1e6, 16)
    acq.set_freqs(f)
    assert acq.freqs.shape == (16,)
    assert acq.time is None

    # Mismatched freq length
    f_bad = np.linspace(0.0, 1e6, 17)
    try:
        acq.set_freqs(f_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched freqs length")


def test_fft_switches_to_freq_domain_and_adds_meta():
    acq_t = _make_small_acq(T=32)
    assert acq_t.mode == "time"
    assert acq_t.time is not None
    assert acq_t.freqs is None

    acq_f = acq_t.fft()
    # Still an AcquisitionData, but in frequency mode
    assert isinstance(acq_f, AcquisitionData)
    assert acq_f.time is None
    assert acq_f.freqs is not None
    assert acq_f.mode == "freqs"

    Tx, Rx, N = acq_t.array.shape
    if np.isrealobj(acq_t.array):
        # rfft path: last axis shrinks
        assert acq_f.array.shape[0:2] == (Tx, Rx)
        assert acq_f.array.shape[2] == acq_f.freqs.size
    else:
        # fft path: length preserved
        assert acq_f.array.shape == (Tx, Rx, N)

    # fft_meta should be present in ctx
    fft_meta = acq_f.ctx.get("fft_meta")
    assert fft_meta is not None
    assert "n_time" in fft_meta and "dt" in fft_meta and "kind" in fft_meta


def test_set_array_and_show_trace_smoke(tmp_path):
    acq = _make_small_acq(T=16)

    new_arr = np.ones_like(acq.array) * 2.0
    acq.set_array(new_arr)
    np.testing.assert_allclose(acq.array, new_arr)

    # Save with explicit path to ensure this branch is covered too
    path = tmp_path / "acq.npz"
    p_out = acq.save(path, compressed=False)
    assert p_out == path
    assert p_out.is_file()

    # Reload and show a trace (Agg backend in conftest)
    acq2 = AcquisitionData.load(p_out)
    ax = acq2.show_trace(0, 0)
    assert ax is not None
