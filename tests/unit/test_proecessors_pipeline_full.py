#!/usr/bin/env python3
import numpy as np

from chirpy.data import AcquisitionData
from chirpy.geometry import GeometryConfigurator
from chirpy.processors import (
    GaussianTimeWindow,
    DTFT,
    AcceptanceMask,
    DownSample,
    MagnitudeOutlierFilter,
    PhaseScreenCorrection,
    Pipeline,
)


def test_full_processor_pipeline_smoke(tiny_grid, ring8, c0):
    """
    End-to-end smoke test for the pure-Python processors on a small
    synthetic acquisition. Stays completely away from k-Wave.
    """
    Tx = ring8.n_tx
    Rx = ring8.n_elements  # full-ring receivers
    T = 64
    t = np.linspace(0.0, 40e-6, T)

    # Simple time-domain data: mixture of low and high frequency sines
    base = 0.5 * np.sin(2 * np.pi * 0.5e5 * t) + 0.1 * np.sin(2 * np.pi * 2.0e5 * t)
    array = np.tile(base, (Tx, Rx, 1))

    acq = AcquisitionData(
        array=array,
        time=t,
        c0=c0,
    )
    geom = GeometryConfigurator(tiny_grid, ring8)
    geom.configure_acceptance(delta=1)
    # geom.select_tx(step=2) # setting only, not change the acq data

    freqs = np.linspace(0.2e5, 0.8e5, 4)

    pipe = Pipeline(
        stages=[
            GaussianTimeWindow(pre_pct=5.0, post_pct=np.inf, c0=c0, geom_config=geom),
            DTFT(freqs),
            PhaseScreenCorrection(geom_config=geom, sign=-1),
            DownSample(geom_config=geom, step=2), # change acq data
            MagnitudeOutlierFilter(geom_config=geom, threshold=0.95),
        ],
        verbose=False,
    )

    pipe(acq, ctx={"tag": "unit-test"})

    # After pipeline:
    #  - data is in freq domain (Tx', Rx, F)
    #  - tx_keep tracks subsampling of tx
    #  - acceptance mask tracks subsampling of rx of each tx
    Tx2, Rx2, F2 = acq.array.shape
    assert Tx2 == len(geom.tx_keep)
    assert Rx2 == ring8.n_elements
    assert F2 == freqs.size

    # Check geom from processors
    # assert "elem_mask" in acq.ctx
    assert geom.elem_mask.shape == (Tx2, Rx2)
    assert "tag" in acq.ctx and acq.ctx["tag"] == "unit-test"

    # No NaNs or infs introduced
    assert np.isfinite(acq.array).all()
