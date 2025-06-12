import numpy as np
from FullWaveUST.geometry.base import Geometry


class ImageGeometry(Geometry):
    """
    Geometry class for 2D image grids. Users can either specify grid parameters
    directly or pass in coordinate arrays.

    Initialization Options (choose one):
      - dx, xmax:
          - dx: grid spacing (float)
          - xmax: half-width of the domain (float); generates a symmetric interval [-xmax, xmax].
          Coordinates are generated automatically in both x and y directions.
      - xi, yi:
          - xi: array of x-coordinates
          - yi: array of y-coordinates

    Attributes
    ----------
    xi : ndarray of shape (Nx,)
        x-direction coordinates
    yi : ndarray of shape (Ny,)
        y-direction coordinates
    """
    def __init__(
        self,
        *,
        dx: float | None = None,
        xmax: float | None = None,
        xi: np.ndarray | None = None,
        yi: np.ndarray | None = None,
    ):
        # Automatically generate coordinates from dx and xmax
        if xi is None:
            if dx is None or xmax is None:
                raise ValueError("Must provide either (dx, xmax) or (xi, yi)")
            # Compute number of grid points and generate symmetric coordinates
            npts = int(round((2 * xmax) / dx)) + 1
            xi = np.linspace(-xmax, xmax, npts)
        xi = np.asarray(xi, dtype=float).ravel()

        # If yi not provided, use a copy of xi
        if yi is None:
            yi = xi.copy()
        else:
            yi = np.asarray(yi, dtype=float).ravel()

        # Compute shape and extent
        nx = xi.size
        ny = yi.size
        extent = (xi.min(), xi.max(), yi.min(), yi.max())
        super().__init__(shape=(ny, nx), extent=extent)

        # Store attributes
        self.xi = xi
        self.yi = yi

    @property
    def nx(self) -> int:
        "Number of grid points in x-direction."
        return self.xi.size

    @property
    def ny(self) -> int:
        "Number of grid points in y-direction."
        return self.yi.size

    @property
    def spacing(self) -> tuple[float, float]:
        "Grid spacing (dx, dy), computed as average difference."
        dx = float(np.mean(np.diff(self.xi))) if self.nx > 1 else 0.0
        dy = float(np.mean(np.diff(self.yi))) if self.ny > 1 else 0.0
        return dx, dy

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return 2D meshgrid arrays (Xi, Yi), useful for plotting or interpolation.
        """
        return np.meshgrid(self.xi, self.yi, indexing='xy')