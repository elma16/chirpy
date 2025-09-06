import numpy as np
from pathlib import Path
from scipy.io import loadmat

from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.data.image_data import ImageData
from chirpy.signals import GaussianModulatedPulse
from chirpy.optimization.operator.wave_operator import WaveOperator
from chirpy.optimization.gradient.time_grad import AdjointStateGrad
from chirpy.optimization.function.least_squares import NonlinearLS

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TKAgg")


ROOT_DIR = Path.cwd().parent
DATA_DIR = Path(ROOT_DIR / "data")
SAVE_DIR = Path("outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

Nx = Ny = 120
dx = dy = 2e-3
f0 = 0.3e6
c0 = 1500.0

n_tx = 256
radius = 110e-3


dfile = SAVE_DIR / "d_obs_for_single.npz"
mat = loadmat(DATA_DIR / "C_true.mat")
model_raw = mat["C_true"]  # (1601,1601)
img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)
img_true = ImageData(model_raw).downsample_to(new_grid=img_grid)
model_true = img_true.array
c_ref = float(model_true.max())

extent = img_grid.extent
c_min = float(model_true.min())
record_time = 1.3 * (extent[1] - extent[0]) / c_min

tx_array = TransducerArray2D.from_ring_array_2D(grid=img_grid, n=n_tx, r=radius)
acq_empty = AcquisitionData.from_geometry(tx_array=tx_array, grid=img_grid)

img = ImageData(array=model_true, grid=img_grid, tx_array=tx_array)
img.show()

pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

if not dfile.exists():
    medium_true = {
        "sound_speed": np.array(model_true, np.float32),
        "density": np.full((Ny, Nx), 1000.0, np.float32),
        "alpha_coeff": np.zeros((Ny, Nx), np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }
    op_true = WaveOperator(
        data=acq_empty,
        medium_params=medium_true,
        record_time=record_time,
        record_full_wf=False,
        use_encoding=False,
        drop_self_rx=True,
        pulse=pulse,
        c_ref=c_ref,
        use_gpu=False,
    )
    acq_sim = op_true.simulate()
    np.savez(dfile, array=acq_sim.array, time=acq_sim.time)

dat = np.load(dfile, allow_pickle=True)
d_obs = dat["array"]
t_vec = dat["time"]
acq_inv = AcquisitionData(array=d_obs, tx_array=tx_array, grid=img_grid, time=t_vec)

model_init = np.full((Ny, Nx), c0, np.float64)

medium_bg = {
    "sound_speed": np.full((Ny, Nx), c0, np.float32),
    "density": np.full((Ny, Nx), 1000.0, np.float32),
    "alpha_coeff": np.zeros((Ny, Nx), np.float32),
    "alpha_power": 1.01,
    "alpha_mode": "no_dispersion",
}

op = WaveOperator(
    data=acq_inv,
    medium_params=medium_bg,
    record_time=record_time,
    record_full_wf=True,
    use_encoding=False,
    drop_self_rx=True,
    pulse=pulse,
    c_ref=c_ref,
    use_gpu=False,
    verbose=False,
)

Fm = op.forward(model_init, kind="c")  # (Tx, n_rx, nt)
D = op.get_field("obs_data")

ge = AdjointStateGrad(op, K=None, seed=0)
r = Fm - D
g = ge.evaluate(model_init, q=r, kind="c")

# fuc = NonlinearLS(op, grad_eval=ge, normalize=True)
# g = fuc.gradient(model_init,kind='c')

np.savez(
    SAVE_DIR / "single_grad_result.npz",
    grad=g,
    residual=r,
    model_init=model_init,
    model_true=model_true,
    time=t_vec,
    dx=dx,
    dy=dy,
)


x_extent = (-Nx / 2 * dx, Nx / 2 * dx)
y_extent = (-Ny / 2 * dy, Ny / 2 * dy)
extent_xy = (x_extent[0], x_extent[1], y_extent[0], y_extent[1])
tx_x, tx_y = op.tx_pos

fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

im0 = ax[0].imshow(model_true, extent=extent_xy, origin="lower", cmap="viridis")
ax[0].set_title("True sound speed [m/s]")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
ax[0].scatter(tx_x, tx_y, marker="*", s=18, color="lime")
fig.colorbar(im0, ax=ax[0], fraction=0.046)

im1 = ax[1].imshow(
    model_init,
    extent=extent_xy,
    origin="lower",
    cmap="viridis",
    vmin=model_true.min(),
    vmax=model_true.max(),
)
ax[1].set_title("Initial model [m/s]")
ax[1].set_xlabel("x [m]")
ax[1].set_ylabel("y [m]")
ax[1].scatter(tx_x, tx_y, marker="*", s=18, color="lime")
fig.colorbar(im1, ax=ax[1], fraction=0.046)

g_abs99 = np.percentile(np.abs(g), 99.0)
im2 = ax[2].imshow(
    g, extent=extent_xy, origin="lower", cmap="seismic", vmin=-g_abs99, vmax=g_abs99
)
ax[2].set_title("Single-shot gradient ∂Φ/∂c")
ax[2].set_xlabel("x [m]")
ax[2].set_ylabel("y [m]")
ax[2].scatter(tx_x, tx_y, marker="*", s=18, color="lime")
fig.colorbar(im2, ax=ax[2], fraction=0.046)

out1 = SAVE_DIR / "single_grad_panels.png"
plt.savefig(out1, dpi=220)


plt.figure(figsize=(6, 5), constrained_layout=True)
plt.imshow(
    g, extent=extent_xy, origin="lower", cmap="seismic", vmin=-g_abs99, vmax=g_abs99
)
plt.scatter(tx_x, tx_y, marker="*", s=18, color="lime")
plt.title("∂Φ/∂c (single, non-encoding)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
out2 = SAVE_DIR / "single_grad_only.png"
plt.colorbar(fraction=0.046)
plt.savefig(out2, dpi=220)
