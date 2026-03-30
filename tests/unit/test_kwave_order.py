from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "chirpy"
    / "optimization"
    / "operator"
    / "_kwave_order.py"
)
_SPEC = importlib.util.spec_from_file_location("chirpy_kwave_order_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

kwave_output_order = _MODULE.kwave_output_order
kwave_source_order = _MODULE.kwave_source_order
mask_linear_indices = _MODULE.mask_linear_indices
reorder_rows_for_mask = _MODULE.reorder_rows_for_mask
reshape_sensor_rows_to_wavefield = _MODULE.reshape_sensor_rows_to_wavefield


def test_kwave_output_order_uses_v061_c_order():
    assert kwave_output_order("0.6.1") == "C"
    assert kwave_output_order("0.6.0") == "F"


def test_kwave_output_order_falls_back_to_docstring():
    doc = "All time-series are (n_sensor, Nt) with sensor points in C-flattened order."
    assert kwave_output_order(None, docstring=doc) == "C"


def test_kwave_source_order_splits_python_and_cpp_at_v061():
    assert kwave_source_order("0.6.1", backend="python") == "C"
    assert kwave_source_order("0.6.1", backend="cpp") == "F"
    assert kwave_source_order("0.6.0", backend="python") == "F"


def test_mask_linear_indices_follow_requested_order():
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(mask_linear_indices(mask, order="C"), [0, 2, 4])
    np.testing.assert_array_equal(mask_linear_indices(mask, order="F"), [0, 3, 4])


def test_reorder_rows_for_mask_f_to_c():
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    rows_f = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.float32)

    rows_c = reorder_rows_for_mask(rows_f, mask, from_order="F", to_order="C")

    np.testing.assert_array_equal(
        rows_c,
        np.array([[10, 11], [30, 31], [20, 21]], dtype=np.float32),
    )


def test_reorder_rows_for_mask_c_to_f():
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    rows_c = np.array([[10, 11], [30, 31], [20, 21]], dtype=np.float32)

    rows_f = reorder_rows_for_mask(rows_c, mask, from_order="C", to_order="F")

    np.testing.assert_array_equal(
        rows_f,
        np.array([[10, 11], [20, 21], [30, 31]], dtype=np.float32),
    )


def test_reshape_sensor_rows_to_wavefield_c_order():
    raw = np.arange(12, dtype=np.float32).reshape(6, 2)
    wf = reshape_sensor_rows_to_wavefield(raw, nt=2, ny=2, nx=3, order="C")

    expected = np.array(
        [
            [[0, 2, 4], [6, 8, 10]],
            [[1, 3, 5], [7, 9, 11]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(wf, expected)


def test_reshape_sensor_rows_to_wavefield_f_order():
    raw = np.arange(12, dtype=np.float32).reshape(6, 2)
    wf = reshape_sensor_rows_to_wavefield(raw, nt=2, ny=2, nx=3, order="F")

    expected = np.array(
        [
            [[0, 4, 8], [2, 6, 10]],
            [[1, 5, 9], [3, 7, 11]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(wf, expected)
