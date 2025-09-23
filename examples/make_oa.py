from pathlib import Path
import h5py
import numpy as np
from scipy.io import savemat
from scipy.ndimage import zoom

from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.optimization.operator.wave_operator import WaveOperator
from chirpy.signals import GaussianModulatedPulse


def load_oa_c_slice(h5_path: Path, z_index: int | None = None):
    with h5py.File(h5_path, "r") as f:
        # pick a dataset that looks like sound speed
        key = None
        for cand in f.keys():
            kl = cand.lower()
            if ("sound" in kl) or ("speed" in kl) or (kl == "c"):
                key = cand
                break
        if key is None:
            # fallback: first 3D float dataset
            for cand in f.keys():
                D = f[cand]
                if (
                    isinstance(D, h5py.Dataset)
                    and D.ndim == 3
                    and np.issubdtype(D.dtype, np.floating)
                ):
                    key = cand
                    break
        if key is None:
            raise RuntimeError("No 3D float dataset found in HDF5 phantom.")

        vol = np.array(f[key])  # unknown axis order
        z_axis = int(np.argmin(vol.shape))  # choose smallest dim as z if ambiguous
        if z_axis != 0:
            vol = np.moveaxis(vol, z_axis, 0)  # (Z,Y,X)

        if z_index is None:
            z_index = vol.shape[0] // 2
        c2d = vol[z_index].astype(np.float64)  # (Y,X)

        # try to read spacing if present (common field names)
        dx = None
        for name in f.keys():
            lname = name.lower()
            if "spacing" in lname or "voxel" in lname or "resolution" in lname:
                sp = np.array(f[name]).astype(float).ravel()
                if sp.size >= 2:
                    dx = float(sp[0])
                    break
        if dx is None:
            dx = 1.0e-3  # 1 mm default

    return c2d, dx


def synthesize_acquisition(
    c2d: np.ndarray,
    dx: float,
    n_tx: int = 128,
    f0: float = 0.3e6,
    ring_margin: float = 0.5,
):
    """
    c2d: (Ny,Nx) m/s, dx in m. ring_margin=0.5 keeps ring well inside box.
    Returns dict with keys: transducerPositionsXY, full_dataset, time, C.
    """
    Ny, Nx = c2d.shape
    # downsample if huge
    target_max = 192
    if max(Ny, Nx) > target_max:
        scale = target_max / max(Ny, Nx)
        Ny_t = int(round(Ny * scale))
        Nx_t = int(round(Nx * scale))
        c2d = zoom(c2d, (Ny_t / Ny, Nx_t / Nx), order=1, prefilter=True)
        Ny, Nx = c2d.shape

    img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)
    xmin, xmax, ymin, ymax = img_grid.extent
    width = xmax - xmin
    height = ymax - ymin
    r = ring_margin * 0.5 * min(width, height)

    ring = TransducerArray2D.from_ring_array_2D(grid=img_grid, n=n_tx, r=r)

    c_min, c_max = float(c2d.min()), float(c2d.max())
    record_time = 1.3 * width / c_min
    pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

    acq_geom = AcquisitionData.from_geometry(grid=img_grid, tx_array=ring, c0=c_max)
    op = WaveOperator(
        data=acq_geom,
        medium_params={"sound_speed": c2d.astype(np.float32)},
        record_time=record_time,
        record_full_wf=False,
        use_encoding=False,
        drop_self_rx=True,
        pulse=pulse,
        c_ref=c_max,
        use_gpu=False,
        verbose=False,
        use_tqdm=True,
        shot_logs=False,
    )
    acq = op.simulate()  # element order (Tx,Rx,T), time axis

    # Build .mat compatible with your frequency-domain example
    raw = {
        "transducerPositionsXY": ring.positions.astype(np.float32),  # (2,N)
        # Your FD script transposes (2,1,0) → we write (T,Rx,Tx) to match that expectation
        "full_dataset": np.transpose(acq.array, (2, 1, 0)).astype(np.float64),
        "time": acq.time.astype(np.float64),
        "C": c2d.astype(np.float32),
    }
    return raw


def main():
    #ROOT_DIR = detect_root()
    H5 = Path(
        "/Users/elliottmacneil/python/msgb/data/NumericalBreastPhantoms-selected/hdf5/Neg_35_Left.h5"
    )  # choose one
    out_mat = Path("/Users/elliottmacneil/python/chirpy/data/kWave_BreastCT.mat")
    out_mat.parent.mkdir(parents=True, exist_ok=True)

    c2d, dx = load_oa_c_slice(H5, z_index=None)
    raw = synthesize_acquisition(c2d, dx, n_tx=120, f0=0.3e6, ring_margin=0.45)
    savemat(out_mat, raw, do_compression=True)
    print(f"[ok] wrote {out_mat}")


if __name__ == "__main__":
    main()
