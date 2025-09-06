# Chirpy 🦜

A Flexible Python Framework for Full-Wave Inversion in Ultrasound Tomography

## Overview

This repository contains the code accompanying the project paper "A Flexible Python Framework for Full-Wave Inversion in Ultrasound Tomography". Chirpy provides a comprehensive toolkit for implementing and experimenting with full-wave inversion techniques in ultrasound tomography applications.

This project is a fork and extension of the original work by Wei Liao ([original repository](https://github.com/weiliao001211/PHAS0077-Research-Project)). 

For time domain simulation, ([k-Wave-python](https://github.com/waltsims/k-wave-python)) is used. This has support for CPU and GPU hardware. For frequency domain simulation on a GPU ([CuPy](https://cupy.dev/)) is used. 


## Installation

### Prerequisites

- Python 3.10 or higher
- For GPU acceleration: CUDA-compatible GPU and drivers

### Basic Installation

Install the core package with CPU-only dependencies:

```bash
git clone https://github.com/[your-username]/chirpy.git
cd chirpy
pip install .
```

### GPU-Accelerated Installation

For frequency domain simulations with GPU acceleration (requires CUDA):

```bash
pip install .[gpu]
```

**Note**: Check the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) to verify hardware compatibility before installing GPU dependencies.

### Development Installation

```bash
pip install .[dev]
```

## References

**[1]** Ali, R., Mitcham, T. M., Brevett, T., Agudo, Ò. C., Martinez, C. D., Li, C., Doyley, M. M., & Duric, N. (2024). 2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver. *IEEE Transactions on Medical Imaging*, 1-1. https://doi.org/10.1109/TMI.2024.3383816

**Associated Code**: https://github.com/rehmanali1994/WaveformInversionUST

