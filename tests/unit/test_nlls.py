#!/usr/bin/env python3
import numpy as np
import pytest

from chirpy.optimization.function import NonlinearLS
from chirpy.optimization.gradient.base import GradientEvaluator
from chirpy.optimization.operator.base import Operator


class DummyOp(Operator):
    """
    Simplest possible Operator with optional encoding hooks.
    """

    def __init__(self, A, d, use_encoding: bool = False):
        self.A = A
        self._d = d
        self.use_encoding = use_encoding
        self.enc_weights = None
        self.enc_delays = None
        self.tau_step = 0
        self.n_tx = A.shape[0]

    def forward(self, m, kind=None):
        return self.A @ m.ravel()

    def get_field(self, key):
        if key == "obs_data":
            return self._d
        raise KeyError(key)

    def renew_encoded_obs(self):
        # No-op for this dummy; NonlinearLS only expects it to exist.
        pass


class DummyGrad(GradientEvaluator):
    """
    Gradient = A^T q with optional encoding metadata.
    """

    def __init__(self, op: DummyOp, K: int | None = None):
        super().__init__(None)
        self.op = op
        self._K = K  # used by NonlinearLS encoding logic
        self._last_q = None
        self._encodings = None
        self._res_cb = None

    def evaluate(self, m, q, kind=None):
        self._last_q = q
        if self._res_cb is not None and q is not None:
            self._res_cb(q)
        return (self.op.A.T @ q).reshape(m.shape)

    def set_residual_callback(self, fn):
        self._res_cb = fn

    # For encoding paths, NonlinearLS looks for these
    def get_last_encodings(self):
        return self._encodings

    def clear_last_encodings(self):
        self._encodings = None


def test_nlls_rejects_nonpositive_scalar_weight():
    ny, nx = 2, 2
    A = np.eye(ny * nx)
    m = np.ones((ny, nx))
    d = np.zeros(ny * nx)
    op = DummyOp(A, d)
    ge = DummyGrad(op)

    with pytest.raises(ValueError):
        NonlinearLS(op, grad_eval=ge, weight=0.0)
    with pytest.raises(ValueError):
        NonlinearLS(op, grad_eval=ge, weight=-1.0)

    # But array-like positive weights are fine
    w_arr = np.ones_like(d)
    fun = NonlinearLS(op, grad_eval=ge, weight=w_arr)
    g = fun.gradient(m)
    assert g.shape == m.shape


def test_nlls_last_misfit_and_value_from_residual():
    ny, nx = 3, 2
    A = np.eye(ny * nx)
    m = np.arange(ny * nx).reshape(ny, nx)
    d = np.zeros(ny * nx)
    op = DummyOp(A, d)
    ge = DummyGrad(op)
    fun = NonlinearLS(op, grad_eval=ge, weight=2.0)

    # Gradient call should set last_misfit
    g = fun.gradient(m)
    assert g.shape == m.shape
    phi = fun.last_misfit
    assert phi > 0

    # value_from_residual should match manual computation
    r = op.forward(m) - d
    q = 2.0 * r
    phi_manual = 0.5 * np.vdot(q, q).real
    assert np.isclose(fun.value_from_residual(r), phi_manual)


def test_nlls_value_uses_cached_residual_when_model_unchanged():
    """
    Check that the cache path in gradient() is exercised without error.
    Hard to assert exact call counts here, but we can at least confirm that
    repeated calls with same model do not change misfit or gradient.
    """
    ny, nx = 2, 3
    A = np.random.randn(5, ny * nx)
    m = np.random.randn(ny, nx)
    d = np.random.randn(5)
    op = DummyOp(A, d)
    ge = DummyGrad(op)
    fun = NonlinearLS(op, grad_eval=ge, weight=1.0)

    g1 = fun.gradient(m)
    phi1 = fun.last_misfit

    g2 = fun.gradient(m.copy())
    phi2 = fun.last_misfit

    np.testing.assert_allclose(g1, g2)
    assert np.isclose(phi1, phi2)
