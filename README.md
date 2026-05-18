# Chirpy 🦜

A Flexible Python Framework for Full-Wave Inversion in Ultrasound Tomography

## Overview

This repository contains the code accompanying the project paper "A Flexible Python Framework for Full-Wave Inversion in Ultrasound Tomography". Chirpy provides a comprehensive toolkit for implementing and experimenting with full-wave inversion techniques in ultrasound tomography applications.

This project is a fork and extension of the original work by Wei Liao ([original repository](https://github.com/weiliao001211/PHAS0077-Research-Project)). 

For time domain simulation, [k-Wave-python](https://github.com/waltsims/k-wave-python) is used. Chirpy targets the unified `kspaceFirstOrder()` API introduced in `k-Wave-python>=0.6.0`, defaulting to the C++ backend while also supporting the NumPy/CuPy solver. Chirpy also normalizes the `v0.6.1` C-order sensor-output change and the remaining backend-specific source-row ordering internally. For frequency domain simulation on a GPU [CuPy](https://cupy.dev/) is used.

## Installation

### Prerequisites

- Python 3.10 or higher
- For GPU acceleration: CUDA-compatible GPU and drivers

### Basic Installation

Install the core package with CPU-only dependencies:

```bash
git clone https://github.com/elma16/chirpy.git
cd chirpy
pip install .
```

Users should also be aware of [some data made available by Wei](https://github.com/weiliao001211/PHAS0077-Research-Project/releases/tag/data).

### GPU-Accelerated Installation

For frequency domain simulations with GPU acceleration (requires CUDA):

```bash
pip install ".[gpu]"
```

**Note**: Check the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) to verify hardware compatibility before installing GPU dependencies.

For time-domain simulations, `WaveOperator` defaults to `kwave_backend="cpp"`. To use the new CuPy-backed k-Wave Python solver on GPU, construct the operator with `kwave_backend="python", use_gpu=True`. Note also that from preliminary use, the CuPy version of k-wave-python is faster than the GPU C++ version only for small to moderate sized grids (64^2 - 128^2). Your mileage may vary.

Chirpy pins `k-wave-python` to the upstream `v0.6.2` tag. That release refreshes the packaged C++ binaries, fixes the C++ source-signal flag handling, and adds first-class `binary_path` support in the unified `kspaceFirstOrder()` API. Chirpy now uses that unified path for both packaged and custom C++ binaries.

### JAX Helmholtz backend

To use the JAX-based Helmholtz solver (`chirpy.optimization.operator.helmholtz_jax`):

```bash
pip install ".[jax_backend]"
```

For numerical parity with SciPy, enable 64-bit by setting `JAX_ENABLE_X64=1` (or calling `jax.config.update("jax_enable_x64", True)` in your code).

### Development Installation

```bash
pip install ".[dev]"
pre-commit install
```

> **⚠️ macOS Users**
>
> `k-wave-python v0.6.2` includes an Apple Silicon OpenMP binary with the HDF5 ABI refresh from [waltsims/k-wave-python#661](https://github.com/waltsims/k-wave-python/issues/661) and the absorbing-media fast-math fix. Chirpy no longer recommends the old Homebrew HDF5 symlink workaround for the packaged binary.
>
> On Intel Macs, upstream currently skips the packaged macOS C++ binary because the v1.4.x Darwin binary is arm64-only. Use the in-process Python solver:
>
> ```python
> op = WaveOperator(..., kwave_backend="python")
> ```
>
> If you want to use a custom C++ backend on macOS, build the Darwin-compatible OpenMP binary and point Chirpy at it explicitly:
>
> ```bash
> git clone https://github.com/elma16/k-wave-omp-darwin.git
> cd k-wave-omp-darwin
> make clean
> make -j"$(sysctl -n hw.logicalcpu)"
> ```
>
> The `Makefile` in that repository links against `/opt/homebrew/opt/hdf5`, so the binary should be built against the HDF5 version actually present on your machine. You can verify the rebuilt binary before using it in Chirpy:
>
> ```bash
> otool -L ./kspaceFirstOrder-OMP | rg hdf5
> ./kspaceFirstOrder-OMP --help
> ```
>
> Once built, point Chirpy’s `WaveOperator` to your compiled `kspaceFirstOrder-OMP` binary from this repository. This custom-binary escape hatch is only supported when `kwave_backend="cpp"`.
> When `binary_path` / `CHIRPY_KWAVE_BIN` is set, Chirpy passes the staged binary to the unified `kspaceFirstOrder(..., binary_path=...)` path and keeps its internal C/F-order normalization for source and sensor rows.
>
> For more details, see: [elma16/k-wave-omp-darwin](https://github.com/elma16/k-wave-omp-darwin).
>
> Chirpy’s test suite treats local HDF5 dylib load failures as environment problems. When that exact macOS load failure is detected, the C++-backend integration tests are skipped instead of failing the whole suite.
>
> **Usage inside Chirpy:** set an environment variable before running examples/tests:
> ```bash
> export CHIRPY_KWAVE_BIN=/path/to/k-wave-omp-darwin/kspaceFirstOrder-OMP
> ```
> When this variable is exported before `pytest`, Chirpy's custom-binary integration fixtures will use it. Those fixtures resolve `CHIRPY_KWAVE_BIN` first, then any `kspaceFirstOrder-OMP` on your `PATH`; the default `WaveOperator(..., kwave_backend="cpp")` path uses the packaged upstream binary when no custom binary is supplied.
> You can quickly verify the binary before running the Chirpy test suite:
> ```bash
> "$CHIRPY_KWAVE_BIN" --help
> ```
> A one-shot test invocation works too if you do not want to export the variable in your shell startup:
> ```bash
> CHIRPY_KWAVE_BIN="$HOME/cpp/k-wave-omp-darwin/kspaceFirstOrder-OMP" pytest -q tests/integration
> ```
> or pass `binary_path=Path("/path/to/kspaceFirstOrder-OMP")` when constructing `WaveOperator`.
>
> Example:
> ```python
> op = WaveOperator(..., kwave_backend="cpp", binary_path=Path("/path/to/kspaceFirstOrder-OMP"))
> ```

### Build the docs locally

Install the docs extras:

```bash
pip install ".[docs]"
```

Build HTML into `docs/_build/html`:

```bash
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to browse the documentation locally.


### Run examples in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/circle_model_td.ipynb
) — **Two-circle TD inversion (toy)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/breast_simulation.ipynb
) — **Breast: Simulation (time-domain)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/breast_time_domain.ipynb
) — **Breast: Inversion (time-domain)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/breast_frequency_domain.ipynb
) — **Breast: Inversion (frequency-domain, Helmholtz)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/elma16/chirpy/blob/main/examples/neural_operator_pipeline.ipynb
) — **Neural Operator Data Gen Pipeline**




## References

**[1]** Ali, R., Mitcham, T. M., Brevett, T., Agudo, Ò. C., Martinez, C. D., Li, C., Doyley, M. M., & Duric, N. (2024). 2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver. *IEEE Transactions on Medical Imaging*, 1-1. https://doi.org/10.1109/TMI.2024.3383816

**Associated Code**: https://github.com/rehmanali1994/WaveformInversionUST
