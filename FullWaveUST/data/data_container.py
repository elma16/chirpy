# FullWaveUST/data/data_container.py

import numpy as np
import matplotlib.pyplot as plt


class DataContainer:
    """
    Base class for data containers: carries a NumPy array and associated geometry.

    Provides common operations such as cloning, I/O, slicing, reshaping,
    padding/cropping, FFT/IFFT, arithmetic, and basic reductions.
    """

    def __init__(self, array: np.ndarray, geometry):
        """
        Parameters
        ----------
        array : np.ndarray
            The raw data array.
        geometry : object
            Geometry object describing the array coordinates.
        ctx : dict
            Context dictionary for additional metadata (default empty).
        """
        self.array = array
        self.geometry = geometry
        self.ctx = {}

    def clone(self):
        """Create a deep copy of this DataContainer, preserving subclass type."""
        return self.__class__(self.array.copy(), self.geometry.copy())

    def copy(self):
        """Alias for clone()."""
        return self.clone()

    def save(self, path: str):
        """
        Save the array to disk.
        If path ends with '.npz', uses numpy.savez; otherwise numpy.save (.npy).
        """
        if path.endswith('.npz'):
            np.savez(path, array=self.array)
        else:
            np.save(path, self.array)

    def visualize(self, **kwargs):
        """
        Display a 2D image of the array using matplotlib, with extent from geometry.
        """
        plt.imshow(self.array, extent=self.geometry.extent, **kwargs)
        plt.title(f"{self.__class__.__name__}")
        plt.colorbar()
        plt.show()

    def reshape(self, new_shape):
        """
        Return a new DataContainer with the array reshaped to new_shape.
        """
        arr = self.array.reshape(new_shape)
        return self.__class__(arr, self.geometry)

    def get_slice(self, index, axis=0):
        """
        Return a slice along the specified axis as a new DataContainer.

        Parameters
        ----------
        index : int
            Index at which to slice.
        axis : int
            Axis along which to slice.
        """
        slicer = [slice(None)] * self.array.ndim
        slicer[axis] = index
        arr = self.array[tuple(slicer)]
        return self.__class__(arr, self.geometry)

    def reorder(self, order):
        """
        Return a new DataContainer with axes permuted according to order.
        """
        arr = np.transpose(self.array, order)
        return self.__class__(arr, self.geometry)

    def pad(self, pad_width, mode='constant', **kwargs):
        """
        Return a new DataContainer with the array padded.

        Parameters
        ----------
        pad_width : sequence of tuple
            Number of values padded to the edges of each axis.
        mode : str
            Padding mode passed to numpy.pad.
        """
        arr = np.pad(self.array, pad_width, mode=mode, **kwargs)
        return self.__class__(arr, self.geometry)

    def crop(self, slices):
        """
        Return a new DataContainer with the array cropped.

        Parameters
        ----------
        slices : tuple of slice
            Slices for each axis, e.g. (slice(a,b), slice(c,d)).
        """
        arr = self.array[slices]
        return self.__class__(arr, self.geometry)

    def fft(self, axis=-1):
        """
        Return a new DataContainer with FFT applied along the given axis.
        """
        arr = np.fft.fft(self.array, axis=axis)
        return self.__class__(arr, self.geometry)

    def ifft(self, axis=-1):
        """
        Return a new DataContainer with IFFT applied along the given axis.
        """
        arr = np.fft.ifft(self.array, axis=axis)
        return self.__class__(arr, self.geometry)

    # Arithmetic operations
    def _binary_op(self, other, op):
        if isinstance(other, DataContainer):
            arr = op(self.array, other.array)
        else:
            arr = op(self.array, other)
        return self.__class__(arr, self.geometry)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __neg__(self):
        return self.__class__(-self.array, self.geometry)

    # Reductions
    def sum(self, axis=None):
        """Return the sum over the specified axis."""
        return self.array.sum(axis=axis)

    def mean(self, axis=None):
        """Return the mean over the specified axis."""
        return self.array.mean(axis=axis)

    def max(self, axis=None):
        """Return the maximum over the specified axis."""
        return self.array.max(axis=axis)

    def min(self, axis=None):
        """Return the minimum over the specified axis."""
        return self.array.min(axis=axis)
