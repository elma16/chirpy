"""
FullWaveUST.processors.time_window
==================================

Apply a "Gaussian-shaped" taper to each time trace to suppress
the signal jumps before and after the geometric time-of-flight (TOF).
"""
from __future__ import annotations

import numpy as np

from .base import BaseProcessor
from ..data import AcquisitionData


class TimeWindow(BaseProcessor):
    """
    Multiply each time trace with an element-specific window
    ::

        w(t,r) = exp(-0.5 * [max(0,  t - TOF)] / τ_post
                    + max(0, TOF - t)] / τ_pre]² )

    where *TOF* is the geometric time-of-flight between transmitter ``Tx`` and
    receiver ``Rx``.  The pre- and post-window half widths are given as a
    percentage of *max(TOF)*, i.e. the longest TOF in the geometry.

    Parameters
    ----------
    pre_pct
        Percent (0–100) of ``max(TOF)`` used for the *leading* half window
        (default 5%).
    post_pct
        Percent of ``max(TOF)`` for the *trailing* half window.  If set to
        ``np.inf`` the trace is not cut at the end (default).
    """

    def __init__(self,
                 pre_pct: float = 5.0,
                 post_pct: float | float("inf") = np.inf) -> None:
        if pre_pct <= 0:
            raise ValueError("`pre_pct` must be positive.")
        self._pre = pre_pct / 100.0
        self._post = float(post_pct)

    @staticmethod
    def _subplus(x: np.ndarray) -> np.ndarray:
        """``max(x, 0)`` implemented with NumPy broadcasting."""
        return np.maximum(x, 0.0)

    # ------------------------------------------------------------------ #
    def __call__(self, data: AcquisitionData) -> None:  # noqa: D401
        """Apply the window to ``data.array`` (shape (Tx, Rx, T)."""
        # get TOF（Rx, Tx）
        tof = data.geometry.compute_geometric_tofs()  # (N, N)
        tau_max = float(tof.max())
        tau_pre = self._pre * tau_max
        tau_post = self._post * tau_max

        # Generate the window for each Tx and multiply it to the original data.
        time_samples = data.time[None, :]  # (1，T)
        n_tx, n_rx, n_t = data.array.shape  # Tx, Rx, T

        for tx in range(n_tx):
            # per-transmitter TOF for all receivers, shape (Rx,)
            tau_tx = tof[:, tx][:, None]  # (Rx, 1)

            # build window of shape (Rx, T)
            w = np.exp(
                -0.5 * (
                        (self._subplus(time_samples - tau_tx) / tau_post)
                        + (self._subplus(tau_tx - time_samples) / tau_pre)
                ) ** 2
            )  # broadcasts (Rx,1) with (1,T) → (Rx,T)

            # multiply in place: data.array[tx] has shape (Rx, T)
            data.array[tx, :, :] *= w.astype(data.array.dtype, copy=False)