"""
FullWaveUST.optimization.function.nonlinear_ls
=============================================

Weighted non-linear least-squares (L2) data misfit.

Objective
---------
    Phi(m) = 0.5 * || w * (F(m) - d) ||^2

where

* F(m): Simulated Tx × Rx frequency-domain data returned by `HelmholtzOperator.forward()`.
* d: Measured data (stored in the operator under key `"obs_data"`).
* w: Optional scalar or array weighting.

Design
------
This class is model-agnostic. All forward modeling is delegated to a `HelmholtzOperator`,
and gradient computations are handled by a user-supplied `GradientEvaluator`
(typically `AdjointHelmholtzGrad`).

It can work with any operator/gradient pair as long as they follow the expected interface.

Internal Caching
----------------
Stores the tuple (model, F(m)) in `self._cache` to avoid redundant forward evaluations
when repeatedly calling `value()`, `gradient()`, or `residual()` at the same model.

Public API
----------
residual(m)
    Return F(m) − d (complex array of shape Tx × Rx); updates cache.
value(m)
    Return scalar misfit Phi(m).
gradient(m)
    Return real-valued gradient image (ny × nx); actual computation is done
    by `grad_eval.gradient_from_q`.

Parameters
----------
op : HelmholtzOperator
    Provides forward modeling and access to measured data.
grad_eval : GradientEvaluator
    Object implementing `gradient_from_q(m, q)` → (ny × nx).
weight : float, ndarray, optional
    A positive scalar or array that scales the residual before applying the L2 norm.

Notes
-----
* Raises `ValueError` if `weight <= 0`.
* The residual weighting `q = w * r` is passed directly to the gradient evaluator,
  ensuring consistency between value and gradient.
"""
from __future__ import annotations
import numpy as np
from typing import Union
from types import SimpleNamespace
from ..operator.helmholtz import HelmholtzOperator
from .base import Function
from ..gradient.base import GradientEvaluator


class NonlinearLS(Function):
    """Weighted L2 misfit wrapper with internal forward-cache."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        op: HelmholtzOperator,
        *,
        grad_eval: GradientEvaluator,
        weight: Union[int, float, np.ndarray] = 1.0,
    ):
        if isinstance(weight, (int, float)) and weight <= 0:
            raise ValueError("weight must be positive")
        self._op, self._ge, self._w = op, grad_eval, weight
        self._cache: SimpleNamespace | None = None   # stores (model, F(m))

    # ------------------------------------------------------------------ #
    def residual(self, m: np.ndarray) -> np.ndarray:
        """Compute and cache r(m) = F(m) − d."""
        Fm = self._op.forward(m)
        self._cache = SimpleNamespace(model=m, F=Fm)
        return Fm - self._op.get_field("obs_data")

    # ------------------------------------------------------------------ #
    def value(self, m: np.ndarray) -> float:
        """Return scalar Φ(m)=½‖w·r‖²."""
        r  = self.residual(m)
        rw = r * self._w
        return 0.5 * np.vdot(rw, rw).real

    # ------------------------------------------------------------------ #
    def gradient(self, m: np.ndarray) -> np.ndarray:
        """
        Return summed gradient image (ny×nx).

        Uses cached F(m) when possible; delegates adjoint work to
        ``grad_eval.gradient_from_q``
        """
        if self._cache is None or not np.array_equal(self._cache.model, m):
            r = self.residual(m)
        else:
            r = self._cache.F - self._op.get_field("obs_data")

        q = r * self._w                      # weight residual
        return self._ge.gradient(m, q)