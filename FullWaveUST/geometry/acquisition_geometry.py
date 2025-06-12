# FullWaveUST/geometry/acquisition_geometry.py

import numpy as np
from FullWaveUST.geometry.base import Geometry


class AcquisitionGeometry(Geometry):
    """
    Geometry for acquisition: transmitter and receiver positions with reference speed.

    Parameters
    ----------
    positions : array_like, shape (2, N)
        (x, y) coordinates of each transducer element.
    c_geom : float
        Reference propagation speed (consistent units).
    """

    def __init__(self, positions, c_geom):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] != 2:
            raise ValueError("positions must be array of shape (2, N)")
        super().__init__(shape=(positions.shape[1],), extent=None)
        self.positions = positions  # shape (2, N)
        self.c_geom = float(c_geom)

    @classmethod
    def from_dict(cls, info):
        """
        Construct AcquisitionGeometry from a dict with keys 'positions' and 'c_geom'.
        """
        return cls(positions=info['positions'], c_geom=info['c_geom'])

    def compute_geometric_tofs(self):
        """
        Compute geometric time-of-flight (TOF) matrix for each tx→rx pair.

        Returns
        -------
        tofs : ndarray, shape (N, N)
            tofs[i, j] = distance from element j to element i divided by c_geom.
        """
        xs, ys = self.positions
        dx = xs[None, :] - xs[:, None]  # shape (N, N)
        dy = ys[None, :] - ys[:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        tofs = dist / self.c_geom  # (Rx, Tx)
        return tofs.T  # (Tx, Rx)

    @property
    def n_elements(self) -> int:
        """Total number of transducer elements (Tx == Rx)."""
        return self.positions.shape[1]
