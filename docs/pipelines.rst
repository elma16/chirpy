Pipelines & Processing
======================

Execution model
---------------

* All processors subclass :class:`chirpy.processors.BaseProcessor` and mutate the same
  :class:`chirpy.data.AcquisitionData` instance in place. No additional copies of the TX×RX tensor
  are created unless a processor explicitly needs one.
* :class:`chirpy.processors.Pipeline` is a thin wrapper that executes a list of processors in order,
  optionally logging timings. You can also inject a ``ctx`` dict at call time to merge ad-hoc
  metadata into ``data.ctx`` before the first stage.

Core processors
---------------

* :class:`chirpy.processors.GaussianTimeWindow` – taper each trace using geometric TOFs to suppress
  pre/post echoes; requires time-axis data.
* :class:`chirpy.processors.DTFT` – discrete-time Fourier transform at user-defined frequencies;
  replaces the time axis with ``freqs`` and converts ``(Tx, Rx, T) → (Tx, Rx, F)``.
* :class:`chirpy.processors.PhaseScreenCorrection` – correct phase error caused by snapping
  transducer coordinates to the reconstruction grid (needs a shared
  :class:`chirpy.geometry.GeometryConfigurator`).
* :class:`chirpy.processors.DownSample` – keep every ``k``-th transmitter and update both geometry
  and acquisition tensor consistently.
* :class:`chirpy.processors.AcceptanceMask` – build a TX×RX boolean mask excluding receivers that
  are too close (circular index distance ≤ ``delta``) and store it under ``data.ctx["elem_mask"]``.
* :class:`chirpy.processors.MagnitudeOutlierFilter` (from ``outlier_removal.py``) – simple Mag/abs
  clipping on spectra to suppress spikes.

Example: time → frequency preprocessing
---------------------------------------

The snippet below mirrors the workflow used in the notebooks: window traces, transform to frequency
domain, correct the grid-induced phase shift, and optionally subsample transmitters for speed.

.. code-block:: python

   import numpy as np
   from chirpy.data import AcquisitionData
   from chirpy.geometry import GeometryConfigurator
   from chirpy.processors import (
       GaussianTimeWindow,
       DTFT,
       PhaseScreenCorrection,
       DownSample,
       MagnitudeOutlierFilter,
       Pipeline,
   )

   # data, grid, tx_array already prepared
   geom_cfg = GeometryConfigurator(grid=data.grid, tx_array=data.tx_array)
   freqs = np.linspace(400e3, 800e3, 5)

   preprocess = Pipeline(
       stages=[
           GaussianTimeWindow(pre_pct=5.0, post_pct=np.inf, geom_config=geom_cfg),
           DTFT(freqs=freqs),
           PhaseScreenCorrection(geom_config=geom_cfg, c0=data.c0),
           DownSample(geom_config=geom_cfg, step=2),         # optional
           MagnitudeOutlierFilter(keep_fraction=0.99),       # optional
       ],
       verbose=True,
   )

   data = preprocess(data)
   print(data.array.shape, data.freqs[:3])

Patterns to keep in mind
------------------------

* All processors mutate ``data``; avoid reassigning ``data`` inside a stage.
* :class:`GeometryConfigurator` is the shared state carrier for anything that needs TX/RX masks,
  snapped coordinates, or acceptance logic—initialise it once and pass it into every relevant
  processor or operator.
* Use ``data.ctx`` for light metadata (e.g., masks, scalar settings). Avoid storing extra copies of
  large arrays there; keep tensors on the main ``data.array`` instead.
