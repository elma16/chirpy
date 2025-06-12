# FullWaveUST/data/acquisition_data.py

import numpy as np
from FullWaveUST.data.data_container import DataContainer


class AcquisitionData(DataContainer):
    """
    Container for acquisition data (raw or preprocessed).

    Attributes
    ----------
    array : np.ndarray
        Data array of shape (Tx,Rx,T) in time-domain or (Tx,Rx,F) in frequency-domain.
    time : np.ndarray
        Time samples vector of length T (None if in frequency-domain).
    geometry : AcquisitionGeometry
        Geometry of transmitters and receivers.
    """

    def __init__(self, array: np.ndarray, geometry, time: np.ndarray | None = None, freqs: np.ndarray | None = None):
        super().__init__(array=array, geometry=geometry)
        self.time = time
        self.freqs = freqs

    def get_array(self):
        """Return the underlying data array."""
        return self.array

    def slice_frequency(self, f_idx: int) -> np.ndarray:
        """
        Return the frequency slice at index f_idx for a frequency-domain array.

        Parameters
        ----------
        f_idx : int
            Frequency index.
        """
        return self.array[:, :, f_idx]
