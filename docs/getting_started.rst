Getting Started
===============

Install
-------

Chirpy targets Python 3.10+. Choose the extra that matches your hardware:

* CPU-only:

  .. code-block:: bash

     pip install .

* GPU acceleration for the Helmholtz (frequency-domain) path:

  .. code-block:: bash

     pip install ".[gpu]"

  Make sure your CUDA toolkit is supported by CuPy (see the `CuPy install guide <https://docs.cupy.dev/en/stable/install.html>`_).

* Development setup:

  .. code-block:: bash

     pip install ".[dev]"
     pre-commit install

macOS C++ backend note
----------------------

Chirpy pins ``k-Wave-python`` to upstream ``v0.6.2``. That release includes an Apple Silicon OpenMP binary with the HDF5 ABI refresh and absorbing-media fast-math fix. On Intel Macs, upstream currently skips the packaged C++ binary because the Darwin v1.4.x binary is arm64-only; use ``kwave_backend="python"`` or build a custom backend.

If you need a custom C++ backend on macOS, build the patched backend:

.. code-block:: bash

   git clone https://github.com/elma16/k-wave-omp-darwin.git
   cd k-wave-omp-darwin
   make clean
   make -j"$(sysctl -n hw.logicalcpu)"

Then point :class:`chirpy.optimization.operator.WaveOperator` at the compiled ``kspaceFirstOrder-OMP`` binary from that repository with ``binary_path`` or ``CHIRPY_KWAVE_BIN``.

Run an example notebook
-----------------------

Quickly explore the library in Colab:

* Two-circle toy inversion (time-domain): https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/circle_model_td.ipynb
* Breast simulation (time-domain): https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/breast_simulation.ipynb
* Breast inversion (time-domain): https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/breast_time_domain.ipynb
* Breast inversion (frequency-domain / Helmholtz): https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/breast_frequency_domain.ipynb
* Neural operator data pipeline: https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/neural_operator_pipeline.ipynb

Small Example
--------------------

This snippet builds a small geometry, creates dummy acquisition data, and runs a preprocessing pipeline:

.. code-block:: python

   import numpy as np

   from chirpy.data import AcquisitionData
   from chirpy.geometry import ImageGrid2D, TransducerArray2D, GeometryConfigurator
   from chirpy.processors import GaussianTimeWindow, DTFT, Pipeline

   # 1) Geometry (centered about 0)
   grid = ImageGrid2D(nx=33, ny=33, dx=5e-4)  # 16 mm field of view

   # simple two-element array on the x-axis
   positions = np.array([[-0.01, 0.01], [0.0, 0.0]])  # shape (2, N)
   is_tx = [True, True]
   is_rx = [True, True]
   tx_array = TransducerArray2D(positions=positions, is_tx=is_tx, is_rx=is_rx)

   # 2) Synthetic time traces (Tx, Rx, T)
   t = np.linspace(0, 6e-4, 256)
   traces = np.sin(2 * np.pi * 600e3 * t)[None, None, :]  # toy pulse
   traces = np.tile(traces, (tx_array.n_tx, tx_array.n_rx, 1))

   data = AcquisitionData(
       traces.astype(np.float32),
       tx_array=tx_array,
       grid=grid,
       time=t,
       c0=1500.0,
   )

   # 3) Preprocess: window then DTFT
   geom_cfg = GeometryConfigurator(grid=grid, tx_array=tx_array)
   pipeline = Pipeline(
       stages=[
           GaussianTimeWindow(pre_pct=5.0, post_pct=np.inf, geom_config=geom_cfg),
           DTFT(freqs=np.array([500e3, 600e3, 700e3])),
       ],
       verbose=True,
   )

   data = pipeline(data)
   print(data.array.shape)  # (Tx, Rx, F)

With the basics confirmed, take a look at the :doc:`data_model` and :doc:`pipelines` pages for the main concepts used throughout the codebase.
