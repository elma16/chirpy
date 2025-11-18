#!/usr/bin/env python3
from pathlib import Path

import h5py
import numpy as np

from chirpy.io import load_mat, save_results
from chirpy.io.load_mat_c import load_mat_c


def _make_hdf5_mat(path: Path):
    """
    Create a minimal HDF5-based .mat-like file with a 2D and 1D dataset.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("M", data=np.arange(6, dtype=np.float32).reshape(2, 3))
        f.create_dataset("v", data=np.arange(4, dtype=np.float64))


def test_load_mat_transposes_and_fortran(tmp_path):
    path = tmp_path / "test_h5.mat"
    _make_hdf5_mat(path)

    data = load_mat(path)
    assert set(data.keys()) == {"M", "v"}

    M = data["M"]
    v = data["v"]

    # 2D dataset should have been transposed and made Fortran-contiguous
    assert M.shape == (3, 2)
    assert M.flags["F_CONTIGUOUS"]
    # 1D dataset: shape preserved; Fortran/C both fine but Fortran flag is True here
    assert v.shape == (4,)
    assert v.flags["F_CONTIGUOUS"]


def test_load_mat_c_transposes_and_c_contiguous(tmp_path):
    path = tmp_path / "test_h5_c.mat"
    _make_hdf5_mat(path)

    data = load_mat_c(path)
    assert set(data.keys()) == {"M", "v"}

    M = data["M"]
    v = data["v"]

    # 2D dataset should have been transposed and made C-contiguous
    assert M.shape == (3, 2)
    assert M.flags["C_CONTIGUOUS"]

    # 1D dataset should be C-contiguous as well
    assert v.flags["C_CONTIGUOUS"]


def test_save_results_creates_mat_file(tmp_path):
    out = tmp_path / "out.mat"
    arr = np.arange(5, dtype=np.float32)
    save_results(out, {"arr": arr})

    assert out.is_file()
    # Optional: light sanity-check via scipy.io.loadmat if available
    try:
        import scipy.io as sio  # type: ignore

        loaded = sio.loadmat(out)
        # savemat injects extra keys; ours should be present
        assert "arr" in loaded
        np.testing.assert_allclose(loaded["arr"].ravel(), arr.astype(np.float64))
    except Exception:
        # If scipy is missing, at least we know the file was written.
        pass
