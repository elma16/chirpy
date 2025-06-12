"""
FullWaveUST.optimization.operator.helmholtz
==========================================

Single-frequency **HelmholtzOperator** that wraps an internal
``functions.HelmholtzSolver`` instance and provides these public services
only:

* :py:meth:`forward(m)`     → F(m)       (Tx × Rx complex)
* :py:meth:`residual(m)`    → F(m) − d
* :py:meth:`backproj(m,q)`  → Jᵀq (ny × nx × Tx real)
* :py:meth:`solve(src, adjoint=False)`
    Thin wrapper around the private solver’s ``solve`` for perturbation
    solves; external code never sees or types the solver.
* :py:meth:`get_field(key)`
    Read-only access to cached arrays/scalars
    (“WF”, “VSRC”, “scaling”, “obs_data”, “PML”, “V”, “freq”).

No other module in the optimisation stack imports **HelmholtzSolver**.
Replacing the underlying solver therefore requires changing *this file
only*.
"""
from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from ...geometry import ImageGeometry
from ...data import AcquisitionData
from .base import Operator
from functions.HelmholtzSolver import HelmholtzSolver          # private

__all__ = ["HelmholtzOperator"]


class HelmholtzOperator(Operator):
    """
    Forward / adjoint operator at a single frequency *f*.

    Parameters
    ----------
    data : AcquisitionData
    f_idx : int
        Index of frequency inside `data`.
    img_geo : ImageGeometry
    sign_conv : {+1, −1}
        iω or −iω convention used throughout.
    a0, L_PML : float
        PML parameters forwarded to HelmholtzSolver.
    """

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        data: AcquisitionData,
        f_idx: int,
        img_geo: ImageGeometry,
        *,
        sign_conv: int,
        a0: float,
        L_PML: float,
    ):
        # --- global parameters & geometry ---------------------------- #
        self._freq  = float(data.freqs[f_idx])
        self._sign  = int(sign_conv)
        self._a0    = float(a0)
        self._L_PML = float(L_PML)

        self._tx_keep = data.ctx.get("tx_keep", np.arange(data.array.shape[0]))
        self._mask    = data.ctx["elem_mask"][self._tx_keep]          # (Tx,Rx)
        self._REC_f   = data.array[self._tx_keep, :, f_idx]           # (Tx,Rx)

        self._x_idx = data.ctx["x_idx"][self._tx_keep]
        self._y_idx = data.ctx["y_idx"][self._tx_keep]
        self._gid   = np.asarray(data.ctx["grid_lin_idx"], np.int64)

        self.ny, self.nx           = img_geo.shape
        self._xi, self._yi         = img_geo.xi, img_geo.yi
        self.n_tx, self.n_rx       = self._REC_f.shape

        # runtime state ----------------------------------------------- #
        self._cache: SimpleNamespace | None = None
        self._atten_phase = False      # set by CG when imag-stage runs

    # ------------------------------------------------------------------ #
    # public helpers
    # ------------------------------------------------------------------ #
    def get_field(self, name: str):
        """
        Read-only access to cached tensors / scalars.

        Legal keys
        ````text
        WF   : forward wavefield            (ny,nx,Tx)
        VSRC : virtual source basis         (ny,nx,Tx)
        scaling : Tx complex scaling vector
        obs_data : measured d               (Tx,Rx)

        PML  : ny×nx real PML weighting
        V    : ny×nx complex slowness
        freq : float centre frequency (Hz)
        ````
        """
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        try:
            return getattr(self._cache, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    def solve(self, src: np.ndarray, *, adjoint: bool = False):
        """Forward wrapper of the internal HelmholtzSolver.solve()."""
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        return self._cache.HS.solve(src, adjoint=adjoint)             # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # basic operator actions
    # ------------------------------------------------------------------ #
    def forward(self, m: np.ndarray) -> np.ndarray:
        """F(m)   Shape (Tx,Rx)."""
        if self._cache is None or not np.array_equal(m, self._cache.model):
            self._build_cache(m)
        return self._extract_sim_data(self._cache.WF)

    def residual(self, m: np.ndarray) -> np.ndarray:
        return self.forward(m) - self._REC_f

    def backproj(self, m: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
        """
        Jᵀq sensitivity cube   Shape (ny,nx,Tx).

        If *q* is None, uses model residual.
        """
        if q is None:
            q = self.residual(m)
        if self._cache is None or not np.array_equal(m, self._cache.model):
            self._build_cache(m)

        WF, VSRC = self._cache.WF, self._cache.VSRC
        scaling = np.zeros((self.n_tx,), np.complex128)
        ADJ_SRC = np.zeros_like(VSRC)

        for s in range(self.n_tx):
            idx = np.where(self._mask[s])[0]
            if idx.size:
                sim = WF[:, :, s].ravel(order="F")[self._gid[idx]]
                scaling[s] = np.vdot(sim, q[s, idx]) / (np.vdot(sim, sim) + 1e-30)
                ys, xs = np.unravel_index(self._gid[idx], (self.ny, self.nx), order="F")
                ADJ_SRC[ys, xs, s] = scaling[s] * sim - q[s, idx]

        self._cache.scaling = scaling
        ADJ_WV, _ = self.solve(ADJ_SRC, adjoint=True)

        bp = np.empty_like(VSRC.real)
        sign = np.sign(self._sign)
        for s in range(self.n_tx):
            vsrc = VSRC[:, :, s] * scaling[s]
            if self._atten_phase:
                vsrc = 1j * sign * vsrc
            bp[:, :, s] = -np.real(np.conj(vsrc) * ADJ_WV[:, :, s])
        return bp

    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    def _build_cache(self, m: np.ndarray) -> None:
        vel   = 1.0 / np.real(m)
        atten = np.sign(self._sign) * np.imag(m) * 2 * np.pi
        HS = HelmholtzSolver(
            self._xi, self._yi, vel, atten,
            self._freq, self._sign, self._a0, self._L_PML,
            canUseGPU=False,
        )

        SRC = np.zeros((self.ny, self.nx, self.n_tx), np.complex128)
        for s, (ix, iy) in enumerate(zip(self._x_idx, self._y_idx)):
            SRC[iy, ix, s] = 1.0
        WF, VSRC = HS.solve(SRC, adjoint=False)

        self._cache = SimpleNamespace(
            model=m.copy(), HS=HS, WF=WF, VSRC=VSRC,
            scaling=None, obs_data=self._REC_f,
            PML=HS.PML, V=HS.V, freq=HS.f
        )

    def _extract_sim_data(self, WF: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_tx, self.n_rx), np.complex128)
        for s in range(self.n_tx):
            idx = self._mask[s]
            out[s, idx] = WF[:, :, s].ravel(order="F")[self._gid[idx]]
        return out