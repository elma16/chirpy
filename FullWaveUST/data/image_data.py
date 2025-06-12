# FullWaveUST/data/image_data.py

import numpy as np
import matplotlib.pyplot as plt
from FullWaveUST.data.data_container import DataContainer


class ImageData(DataContainer):
    """
    Container for image data (e.g., reconstructed slowness).

    Attributes
    ----------
    array : np.ndarray
        2D image array of shape (Ny, Nx).
    geometry : ImageGeometry
        Geometry of the image grid.
    history : list of np.ndarray
        Snapshots of the image at each update.
    """

    def __init__(self, array: np.ndarray, geometry):
        super().__init__(array=array, geometry=geometry)
        self.history = [array.copy()]

    @property
    def current(self) -> np.ndarray:
        """Return the current image array."""
        return self.array

    def update(self, new_array: np.ndarray):
        """
        Update the current image array and append to history.

        Parameters
        ----------
        new_array : np.ndarray
            Updated image array.
        """
        self.array = new_array.copy()
        self.history.append(self.array)

    def visualize(self, idx: int = -1, **kwargs):
        """
        Visualize a saved history frame.

        Parameters
        ----------
        idx : int
            Index into history list (default: last).
        """
        img = self.history[idx]
        plt.imshow(img, extent=self.geometry.extent, **kwargs)
        plt.title(f"ImageData iteration {idx}")
        plt.colorbar()
        plt.show()

    def save(self, path: str):
        """
        Save current image and full history to a .npz file.

        Parameters
        ----------
        path : str
            File path ending with .npz.
        """
        np.savez(path, current=self.array, history=np.stack(self.history))
