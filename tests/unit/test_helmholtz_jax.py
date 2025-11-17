#!/usr/bin/env python3
"""
Unit tests for the JAX Helmholtz solver.

Covers:
- Multi-source vmap behaviour (batch vs per-source solves).
- Linearity of the inverse operator (forward solve).
- Differentiability w.r.t. sources (grad through block-LU solve).
- Smoke test for forward + adjoint stack.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# Run tests in x64 to match SciPy-comparison numerics
jax.config.update("jax_enable_x64", True)

from chirpy.optimization.operator.helmholtz_jax import (  # noqa: E402
    HelmholtzSolverJAX,
    _applyBlockLU_jax_single,
)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


def _make_small_solver() -> HelmholtzSolverJAX:
    """
    Construct a small, cheap HelmholtzSolverJAX instance for tests.

    Grid: 24x24, homogeneous velocity, zero attenuation.
    """
    Nx = Ny = 24
    dx = dy = 1.0e-3  # 1 mm
    x = np.linspace(0.0, (Nx - 1) * dx, Nx)
    y = np.linspace(0.0, (Ny - 1) * dy, Ny)

    vel = np.full((Ny, Nx), 1500.0, dtype=np.float64)
    atten = np.zeros_like(vel)

    f = 0.5e6  # 0.5 MHz
    sign_conv = -1
    a0 = 10.0
    L_pml = 9.0e-3

    solver = HelmholtzSolverJAX(
        x=x,
        y=y,
        vel=vel,
        atten=atten,
        f=f,
        signConvention=sign_conv,
        a0=a0,
        L_PML=L_pml,
    )
    return solver


def _rand_complex(key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    """Utility to generate a random complex array with unit-ish scale."""
    k1, k2 = jax.random.split(key)
    real = jax.random.normal(k1, shape)
    imag = jax.random.normal(k2, shape)
    return real + 1j * imag


# -----------------------------------------------------------------------------#
# Tests
# -----------------------------------------------------------------------------#


def test_vmapped_multiple_sources_matches_per_source():
    """
    Check that solving K sources in batch via vmap matches
    solving each source individually and stacking.

    We check:
      - wavefield wv: strict absolute tolerance
      - virtual source virt: relative tolerance (since it's scaled by ~f^2)
    """
    solver = _make_small_solver()
    Ny, Nx = solver.Ny, solver.Nx
    K = 4

    key = jax.random.PRNGKey(0)
    src_jax = _rand_complex(key, (Ny, Nx, K))
    src = np.array(src_jax, dtype=np.complex128)

    # Batched solve
    wv_batch, virt_batch = solver.solve(src, adjoint=False)
    assert wv_batch.shape == (Ny, Nx, K)
    assert virt_batch.shape == (Ny, Nx, K)

    # Per-source solve
    wv_list = []
    virt_list = []
    for k in range(K):
        src_k = src[..., k : k + 1]  # (Ny, Nx, 1)
        wv_k, virt_k = solver.solve(src_k, adjoint=False)
        wv_list.append(wv_k[..., 0])
        virt_list.append(virt_k[..., 0])

    wv_stack = np.stack(wv_list, axis=-1)
    virt_stack = np.stack(virt_list, axis=-1)

    # Wavefield: absolute max diff should be tiny
    max_diff_wv = np.max(np.abs(wv_batch - wv_stack))
    assert max_diff_wv < 1e-9

    # Virtual source: use relative tolerance due to huge scaling (sf ~ f^2)
    num = np.max(np.abs(virt_batch - virt_stack))
    den = np.max(np.abs(virt_stack)) + 1e-15
    rel = num / den
    assert rel < 1e-10, f"Virtual field batch vs per-source rel diff={rel:.3e}"


def test_linearity_of_inverse():
    """
    Check linearity: H^{-1}(a s1 + b s2) ≈ a H^{-1} s1 + b H^{-1} s2
    via the forward solve for the inverse.
    """
    solver = _make_small_solver()
    Ny, Nx = solver.Ny, solver.Nx
    K = 1

    key = jax.random.PRNGKey(1)
    s1 = np.array(_rand_complex(key, (Ny, Nx, K)), dtype=np.complex128)
    s2 = np.array(
        _rand_complex(jax.random.split(key)[0], (Ny, Nx, K)), dtype=np.complex128
    )

    a = 1.3 + 0.7j
    b = -0.4 + 2.1j

    # Solve independently
    u1, _ = solver.solve(s1, adjoint=False)
    u2, _ = solver.solve(s2, adjoint=False)

    # Solve combined
    s_comb = a * s1 + b * s2
    u_comb, _ = solver.solve(s_comb, adjoint=False)

    # Compare u_comb vs a*u1 + b*u2
    u_lin = a * u1 + b * u2

    num = np.max(np.abs(u_comb - u_lin))
    den = np.max(np.abs(u_lin)) + 1e-15
    rel = num / den

    assert rel < 1e-8, f"Linearity violated: rel={rel:.3e}, max|Δ|={num:.3e}"


def test_differentiable_wrt_source():
    """
    Check that the single-source block-LU solve is differentiable w.r.t. the source.

    We differentiate a simple scalar functional of the solution:
        L(s) = ||u||^2
    where u = H^{-1} s (via _applyBlockLU_jax_single).
    """
    solver = _make_small_solver()
    factors = solver.factors
    Nx = solver.Nx
    Ny = solver.Ny

    key = jax.random.PRNGKey(3)
    src0 = _rand_complex(key, (Nx, Ny))  # (Nx, Ny) for _applyBlockLU_jax_single

    # Represent complex src as real 2x(Nx,Ny) for real-valued grad
    src_init = jnp.stack([jnp.real(src0), jnp.imag(src0)], axis=0)

    def loss_fn(src_ri: jnp.ndarray) -> jnp.ndarray:
        """
        src_ri: shape (2, Nx, Ny), where
            src_ri[0] = Re(s), src_ri[1] = Im(s)
        """
        s_c = src_ri[0] + 1j * src_ri[1]
        u = _applyBlockLU_jax_single(s_c, factors, adjoint=False)
        # Simple real scalar functional
        return jnp.real(jnp.vdot(u, u))

    # jax.grad should run without error and return finite values
    grad_src = jax.grad(loss_fn)(src_init)

    assert grad_src.shape == src_init.shape
    assert jnp.all(jnp.isfinite(grad_src)), "Non-finite gradients found in dL/ds"


@pytest.mark.slow
def test_full_stack_smoke():
    """
    Smoke test: run forward and adjoint solves on multiple sources,
    just to ensure nothing obvious explodes when composed.

    NOTE: This does NOT assert the exact algebraic adjointness of the
    current adjoint implementation; that needs a separate audit against
    the original MATLAB/CuPy adjoint solver.
    """
    solver = _make_small_solver()
    Ny, Nx = solver.Ny, solver.Nx
    K = 3

    key = jax.random.PRNGKey(4)
    src = np.array(_rand_complex(key, (Ny, Nx, K)), dtype=np.complex128)

    wv_fwd, virt_fwd = solver.solve(src, adjoint=False)
    wv_adj, virt_adj = solver.solve(src, adjoint=True)

    assert wv_fwd.shape == (Ny, Nx, K)
    assert virt_fwd.shape == (Ny, Nx, K)
    assert wv_adj.shape == (Ny, Nx, K)
    assert virt_adj.shape == (Ny, Nx, K)

    # Basic sanity: no NaNs or infs
    assert np.all(np.isfinite(wv_fwd))
    assert np.all(np.isfinite(virt_fwd))
    assert np.all(np.isfinite(wv_adj))
    assert np.all(np.isfinite(virt_adj))
