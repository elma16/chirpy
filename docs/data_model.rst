Data & Geometry
===============

Containers
----------

.. class:: chirpy.data.DataContainer

   Base class that pairs a NumPy array with optional geometry (:class:`chirpy.geometry.ImageGrid2D`,
   :class:`chirpy.geometry.TransducerArray2D`) and a free-form context dictionary ``ctx`` for
   metadata. It supports light-weight array-like operations (reshape, reorder, pad, crop, FFT/ifft)
   that preserve geometry + context, plus simple ``save`` / ``load`` helpers for ``.npy``/``.npz``.


.. class:: chirpy.data.AcquisitionData

   Specialised container for measured/simulated TX→RX datasets. Expected shapes are ``(Tx, Rx, T)``
   for time traces or ``(Tx, Rx, F)`` for spectra. You supply either ``time`` **or** ``freqs`` to
   describe the third axis, along with the matching transducer array and image grid. Convenience
   methods:

   * :py:meth:`AcquisitionData.from_geometry` builds an empty container with only geometry/axes.
   * :py:meth:`AcquisitionData.set_time` / :py:meth:`set_freqs` keep the axes in sync with a
     pre-loaded array.
   * :py:meth:`slice_frequency` returns a ``(Tx, Rx)`` slice for a specific ``f_idx`` when working
     in the frequency domain.
   * :py:meth:`save` serialises the array, geometry, axes, and ``ctx`` to ``.npz`` without pickle.

Minimal construction example:

.. code-block:: python

   import numpy as np
   from chirpy.data import AcquisitionData
   from chirpy.geometry import ImageGrid2D, TransducerArray2D

   grid = ImageGrid2D(nx=128, ny=128, dx=5e-4)
   positions = np.stack([np.cos(np.linspace(0, 2*np.pi, 32, endpoint=False)) * 0.04,
                         np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False)) * 0.04])
   roles = dict(is_tx=np.ones(32, bool), is_rx=np.ones(32, bool))
   tx_array = TransducerArray2D(positions=positions, **roles)

   t = np.linspace(0, 8e-4, 1024)
   traces = np.zeros((tx_array.n_tx, tx_array.n_rx, t.size), dtype=np.float32)

   data = AcquisitionData(traces, tx_array=tx_array, grid=grid, time=t, c0=1500.0)
   data.ctx["comment"] = "empty shell to be filled by simulator"

Geometry helpers
----------------

.. class:: chirpy.geometry.ImageGrid2D

   Centred 2-D grid for spatial discretisation. You can pass explicit coordinates (``xi``, ``yi``),
   uniform dimensions (``nx``, ``ny``, ``dx``/``dy``), or half-widths (``dx``, ``xmax``/``ymax``),
   and it will build k-Wave-style axes around zero. Utility methods include ``coord2index``,
   ``max_f`` (spatial Nyquist check), and ``kmesh`` for wavenumber grids.

.. class:: chirpy.geometry.TransducerArray2D

   Describes the ring/array of elements with TX/RX roles. Construct with a list of
   :class:`chirpy.geometry.Transducer` instances or raw ``positions`` plus ``is_tx`` / ``is_rx``
   flags. Supports element-level RX gating (``set_rx_flags``), geometry-aware TOF estimates
   (``geometric_tofs``), and receiver masks on a grid (``get_rx_mask*``).

.. class:: chirpy.geometry.GeometryConfigurator
   :no-index:

   Utility that maps continuous transducer positions onto an :class:`ImageGrid2D`, tracks active TX
   / RX subsets, and builds TX×RX acceptance masks (either circular ``delta`` exclusion or
   user-provided masks). Operators and processors reuse it to stay in sync with geometry during
   simulation/inversion.

I/O utilities
-------------

* :func:`chirpy.io.load_mat.load_mat` and :func:`chirpy.io.load_mat_c.load_mat_c` load MATLAB ``.mat``
  files into NumPy arrays (transposing to match MATLAB ordering). Progress and shapes are printed so
  large files are traceable.
* :func:`chirpy.io.save_results.save_results` writes a dictionary of arrays back to ``.mat`` with
  compression enabled.

Plotting helpers
----------------

* :py:meth:`chirpy.data.DataContainer.show` provides a quick ``matplotlib`` view for 1-D or 2-D
  arrays, preserving grid extents when available.
* :mod:`chirpy.utils.animate_results` and :mod:`chirpy.utils.visualizer_multi_mode` offer
  convenience routines to visualise reconstructions and multichannel outputs across time/frequency.
