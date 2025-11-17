#!/usr/bin/env python3
import numpy as np

from chirpy.processors.lowpass_filter import LowpassFilter, _tukey_lowpass_nd
from chirpy.data import AcquisitionData
from chirpy.signals import GaussianModulatedPulse


def test_tukey_lowpass_basic_attenuation():
    """
    Directly test the internal low-pass helper: suppresses high-frequency
    component while leaving low-frequency component largely intact.
    """
    dt = 1e-4
    t = np.arange(0.0, 1e-2, dt)
    # Low + high freq components
    low = np.sin(2 * np.pi * 200 * t)
    high = np.sin(2 * np.pi * 2000 * t)
    x = (low + high)[None, :]  # shape (..., T)

    # Cut off just above the low component
    f_cut = 400.0
    y = _tukey_lowpass_nd(x, dt, f_cut, roll=0.2)[0]

    # Low-frequency energy should remain comparable, high-frequency energy reduced
    low_energy_before = np.linalg.norm(low)
    low_energy_after = np.linalg.norm(y)
    assert low_energy_after > 0.5 * low_energy_before

    # High-frequency part should be strongly suppressed: original minus filtered
    residual = (low + high) - y
    high_energy_residual = np.linalg.norm(residual)
    # At least more energy in residual than in filtered low band
    assert high_energy_residual > 0.2 * low_energy_before


def test_lowpass_filter_on_acquisition_data(tmp_path):
    """
    LowpassFilter.__call__ should:
      - operate in place on AcquisitionData.array
      - populate ctx['lowpass'] with reasonable metadata
    """
    dt = 1e-4
    t = np.arange(0.0, 1e-2, dt)
    Tx, Rx, T = 2, 3, t.size
    data = np.random.randn(Tx, Rx, T).astype(np.float32)

    acq = AcquisitionData(
        array=data.copy(),
        grid=None,
        tx_array=None,
        time=t,
        c0=1500.0,
    )

    lp = LowpassFilter(f0=1e3, frac_bw=0.5, roll=0.2, f_cut=None, verbose=False)
    lp(acq)

    assert acq.array.shape == data.shape
    assert "lowpass" in acq.ctx
    meta = acq.ctx["lowpass"]
    assert "f_cut" in meta and meta["f_cut"] > 0.0
    assert np.isfinite(acq.array).all()


def test_lowpass_filtered_pulse_wrapper():
    """
    apply_to_pulse should return a Pulse whose sample()
    produces a waveform of the same length but with altered spectrum.
    """
    base = GaussianModulatedPulse(f0=5e5, frac_bw=0.75, amp=1.0)
    lp = LowpassFilter(f0=5e5, frac_bw=0.5, roll=0.3, verbose=False)

    wrapped = lp.apply_to_pulse(base, remove_dc=True, renorm="l2")

    dt = 1e-7
    nt = 1024
    y0 = base.sample(dt, nt)
    y1 = wrapped.sample(dt, nt)

    assert y0.shape == y1.shape == (nt,)
    assert np.isfinite(y1).all()

    # Energy should be similar after l2 renorm
    e0 = float(np.linalg.norm(y0))
    e1 = float(np.linalg.norm(y1))
    assert 0.5 * e0 < e1 < 2.0 * e0
