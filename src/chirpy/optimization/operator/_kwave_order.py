from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Literal

import numpy as np

ArrayOrder = Literal["C", "F"]


def resolve_kwave_version(module) -> str | None:
    """Best-effort version lookup for the installed kwave package."""
    version = getattr(module, "__version__", None) or getattr(module, "VERSION", None)
    if version is not None:
        return str(version)

    for dist_name in ("k-Wave-python", "k-wave-python", "kwave"):
        try:
            return package_version(dist_name)
        except PackageNotFoundError:
            continue
    return None


def parse_kwave_version(version: str | None) -> tuple[int, int, int]:
    if not version:
        return (0, 0, 0)
    parts = [int(part) for part in re.findall(r"\d+", str(version))]
    if not parts:
        return (0, 0, 0)
    parts.extend([0] * max(0, 3 - len(parts)))
    return tuple(parts[:3])


def kwave_output_order(
    version: str | None, *, docstring: str | None = None
) -> ArrayOrder:
    """
    Sensor output order used by the unified kspaceFirstOrder API.

    k-Wave-python v0.6.1 switched unified solver outputs to C-order.
    """
    if parse_kwave_version(version) >= (0, 6, 1):
        return "C"
    if docstring and "C-flattened order" in docstring:
        return "C"
    return "F"


def kwave_source_order(
    version: str | None,
    *,
    backend: str,
    docstring: str | None = None,
) -> ArrayOrder:
    """
    Multi-row source order expected by the backend boundary.

    v0.6.1 changed the in-process Python solver to C-order, but the C++
    backend still serializes binary masks through the legacy Fortran-ordered
    HDF5 boundary.
    """
    if backend == "python" and kwave_output_order(version, docstring=docstring) == "C":
        return "C"
    return "F"


def linear_index_2d(ix: int, iy: int, *, ny: int, nx: int, order: ArrayOrder) -> int:
    if order == "C":
        return iy * nx + ix
    return iy + ny * ix


def mask_linear_indices(mask: np.ndarray, *, order: ArrayOrder) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool).ravel(order=order))


def reorder_rows_for_mask(
    rows: np.ndarray,
    mask: np.ndarray,
    *,
    from_order: ArrayOrder,
    to_order: ArrayOrder,
) -> np.ndarray:
    """
    Reorder per-point rows between C- and F-flattened traversals of a binary mask.
    """
    arr = np.asarray(rows)
    if arr.ndim < 2:
        raise ValueError(f"Expected row-major source data with ndim>=2, got {arr.shape}")
    if arr.shape[0] <= 1 or from_order == to_order:
        return arr

    mask_arr = np.asarray(mask, dtype=bool)
    from_lin = mask_linear_indices(mask_arr, order=from_order)
    to_lin = mask_linear_indices(mask_arr, order=to_order)
    if from_lin.size != arr.shape[0]:
        raise ValueError(
            "Row count does not match the number of active points in the source mask: "
            f"{arr.shape[0]} rows vs {from_lin.size} active mask points"
        )

    from_coords = np.column_stack(
        np.unravel_index(from_lin, mask_arr.shape, order=from_order)
    )
    to_coords = np.column_stack(
        np.unravel_index(to_lin, mask_arr.shape, order=to_order)
    )
    from_lookup = {tuple(coord): idx for idx, coord in enumerate(from_coords.tolist())}
    perm = np.array([from_lookup[tuple(coord)] for coord in to_coords.tolist()])
    return arr[perm]


def reshape_sensor_rows_to_wavefield(
    raw: np.ndarray,
    *,
    nt: int,
    ny: int,
    nx: int,
    order: ArrayOrder,
) -> np.ndarray:
    arr = np.asarray(raw)
    expected_shape = (ny * nx, nt)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Expected full-grid sensor data with shape {expected_shape}, got {arr.shape}"
        )
    return arr.reshape((ny, nx, nt), order=order).transpose(2, 0, 1)
