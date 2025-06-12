from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use an interactive backend and enable interactive mode
matplotlib.use("TkAgg")
plt.ion()

# -------------------- unit conversions ------------------------------- #
_NP2DB    = 20.0 / np.log(10.0)    # 1 Np → dB
_SLOW2ATT = 1e6 / 1e2              # (Hz·m)⁻¹ → (MHz·cm)⁻¹


class InversionVisualizer:
    """
    Simple live viewer; call :meth:`update` after each CG iteration.

    Parameters
    ----------
    xi, yi : ndarray
        Imaging grid (1-D). Only used for axis extent.
    C_true : ndarray
        Ground-truth velocity for reference (Ny×Nx).
    atten_true : ndarray
        Ground-truth attenuation in **dB/cm**.
    sign_conv : int, optional
        Helmholtz sign convention used upstream (±1). Default −1.
        If provided, estimated attenuation will be multiplied by
        ``sign_conv`` so that a *larger* α appears *brighter*.
    """

    def __init__(
        self,
        xi: np.ndarray,
        yi: np.ndarray,
        C_true: np.ndarray,
        atten_true: np.ndarray,         # already in dB/cm
        *,
        sign_conv: int = -1,
    ) -> None:
        Ny, Nx = C_true.shape
        self.xi, self.yi = xi, yi
        self._sign_conv = int(np.sign(sign_conv)) or 1

        # global colour limits
        vmin_true, vmax_true = float(C_true.min()), float(C_true.max())
        atten_range = 10 * np.array([-1, 1], float)  # ±10 dB/cm

        # -------- figure scaffold ------------------------------------- #
        self.fig, self.axes = plt.subplots(2, 3, figsize=(9, 6))

        # 0,0 Estimated velocity
        self.im_est_vel = self.axes[0, 0].imshow(
            np.zeros((Ny, Nx)), cmap="gray", aspect="auto",
            extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        )
        self.axes[0, 0].set_title("Estimated Wave Velocity")

        # 0,1 True velocity
        self.axes[0, 1].imshow(
            C_true, vmin=vmin_true, vmax=vmax_true,
            cmap="gray", aspect="auto",
            extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        )
        self.axes[0, 1].set_title("True Wave Velocity")

        # 0,2 Search direction
        self.im_search = self.axes[0, 2].imshow(
            np.zeros((Ny, Nx)), cmap="gray", aspect="auto",
            extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        )
        self.axes[0, 2].set_title("Search Direction")

        # 1,0 Estimated attenuation
        self.im_est_atten = self.axes[1, 0].imshow(
            np.zeros((Ny, Nx)), vmin=atten_range[0], vmax=atten_range[1],
            cmap="gray", aspect="auto",
            extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        )
        self.axes[1, 0].set_title("Estimated Attenuation")

        # 1,1 True attenuation
        self.axes[1, 1].imshow(
            atten_true, vmin=atten_range[0], vmax=atten_range[1],
            cmap="gray", aspect="auto",
            extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        )
        self.axes[1, 1].set_title("True Attenuation")

        # 1,2 Gradient
        self.im_gradient = self.axes[1, 2].imshow(
            np.zeros((Ny, Nx)), cmap="gray", aspect="auto",
            extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        )
        self.axes[1, 2].set_title("Gradient")

        plt.tight_layout()
        plt.pause(0.01)
        self.global_iter = 0

    def _atten_dbcm_from_slow(self, slow: np.ndarray) -> np.ndarray:
        """
        Convert Im(slow) → α[dB/cm] and ensure **higher α = brighter**.
        """
        alpha = _NP2DB * _SLOW2ATT * (2.0 * np.pi * np.imag(slow))
        return self._sign_conv * alpha

    @staticmethod
    def _safe_clim(img: np.ndarray, pad: float = 1e-6) -> tuple[float, float]:
        vmin, vmax = float(img.min()), float(img.max())
        if vmin == vmax:
            vmin -= pad
            vmax += pad
        return vmin, vmax

    def update(
        self,
        slow: np.ndarray,
        gradient: np.ndarray,
        search_dir: np.ndarray,
    ) -> None:
        """Refresh all subplots after each iteration."""
        self.global_iter += 1
        k = self.global_iter

        # Estimated velocity
        v = 1.0 / np.real(slow)
        self.im_est_vel.set_data(v)
        self.im_est_vel.set_clim(*self._safe_clim(v))
        self.axes[0, 0].set_title(f"Estimated Wave Velocity {k}")

        # Estimated attenuation
        a = self._atten_dbcm_from_slow(slow)
        self.im_est_atten.set_data(a)
        self.axes[1, 0].set_title(f"Estimated Attenuation {k}")

        # Search direction
        sd = search_dir.real
        self.im_search.set_data(sd)
        self.im_search.set_clim(*self._safe_clim(sd))
        self.axes[0, 2].set_title(f"Search Direction {k}")

        # Gradient
        g = -np.real(gradient)
        self.im_gradient.set_data(g)
        self.im_gradient.set_clim(*self._safe_clim(g))
        self.axes[1, 2].set_title(f"Gradient Iteration {k}")

        # Force immediate redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)