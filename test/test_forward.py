import numpy as np
import matplotlib.pyplot as plt

from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.optimization.operator.wave_operator import WaveOperator
from chirpy.signals import GaussianModulatedPulse
from chirpy.data import ImageData

import matplotlib

# Use TkAgg backend for interactive plotting
matplotlib.use("TkAgg")

# ----------------------------
# 1) Grid & true model
# ----------------------------
Nx = Ny = 240
dx = dy = 1e-3
c0 = 1500.0
record_time = 1.2 * Nx * dx / c0
n_tx = 64
f0 = 0.3e6

# Imaging grid (centered about 0)
img_grid = ImageGrid2D(dx=dx, nx=Nx, ny=Ny)

# Background & anomalous true model
model_bg = np.full((Ny, Nx), c0, np.float64)

# Use XY with 'xy' indexing so arrays are (ny, nx)
X, Y = img_grid.meshgrid(indexing="xy")

model_true = model_bg.copy()
model_true[((X - 0.03) ** 2 + (Y - 0.03) ** 2) < 0.012**2] = c0 + 200  # faster blob
model_true[((X + 0.02) ** 2 + (Y + 0.016) ** 2) < 0.010**2] = c0 - 200  # slower blob

# ----------------------------
# 2) Build a 64-element ring
# ----------------------------
ring = TransducerArray2D.from_ring_array_2D(
    grid=img_grid,
    r=(min(Nx, Ny) // 2 - 10) * dx,  # radius inside grid
    n=n_tx,
)

# Pick "top" (max y) as TX, and "bottom" (min y) as RX
pos = ring.positions  # shape (2, N)
top_idx = int(np.argmax(pos[1]))  # highest y
bot_idx = int(np.argmin(pos[1]))  # lowest y

tx_pos = pos[:, top_idx].reshape(2, 1)
rx_pos = pos[:, bot_idx].reshape(2, 1)

# Build a new 2-element array: one TX at top, one RX at bottom
pair_positions = np.hstack([tx_pos, rx_pos])  # (2, 2)
is_tx = np.array([True, False], dtype=bool)
is_rx = np.array([False, True], dtype=bool)
pair_array = TransducerArray2D(positions=pair_positions, is_tx=is_tx, is_rx=is_rx)
print(pair_array.transducers)
img = ImageData(array=model_true, grid=img_grid, tx_array=pair_array)
img.show()

# Acquisition container using the 1-Tx / 1-Rx pair
acq_pair = AcquisitionData.from_geometry(grid=img_grid, tx_array=pair_array)

# ----------------------------
# 3) Forward operator
# ----------------------------
medium_params = {
    "sound_speed": model_true.astype(np.float32),  # k-Wave expects float32
    # (density/alpha can be omitted -> defaults)
}

pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

op = WaveOperator(
    data=acq_pair,
    medium_params=medium_params,
    record_time=record_time,
    pulse=pulse,
    use_encoding=False,
    drop_self_rx=False,
    record_full_wf=True,  # <-- record the full wavefield for snapshots
    cfl=0.2,
    c_ref=c0,
    pml_size=10,
    pml_alpha=10.0,
    scale_source_terms=True,
    src_mode="additive",
    pml_inside=False,
    use_gpu=False,  # set True if your kwave C GPU backend is available
    verbose=False,
)

# ----------------------------
# 4) One forward shot
# ----------------------------
Fm = op.forward(model_true.astype(np.float32), kind="c")  # (Tx=1, n_rx=1, nt)
t = op.time_axis  # (nt,)
WF = op.get_forward_fields()[0]  # (nt, ny, nx) for our single-Tx

trace = Fm[0, 0]  # the only Rx trace, shape (nt,)

from matplotlib import colors  # NEW

# ----------------------------
# 5) Pick 5 times & plot
# ----------------------------
# Choose 5 (here 8) time instants across the record (10%..90%)
n = 6
t_sel = np.linspace(0.1, 0.9, n) * t[-1]
idx_sel = [int(np.argmin(np.abs(t - ts))) for ts in t_sel]

# ---- NEW: global symmetric normalization (0 is center) ----
WF_pool = WF
# Robust upper limit
A = np.percentile(np.abs(WF_pool), 99.5)
norm = colors.TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=A)

# Figure: top row = full trace with red vertical lines
#         bottom row = wavefield snapshots (shared vmin/vmax via `norm`)
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(2, n, height_ratios=[1, 2], hspace=0.35)

# Top row
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(t * 1e3, trace, lw=1.0)
for ts in t_sel:
    ax0.axvline(ts * 1e3, color="red", linestyle="--", linewidth=1)
ax0.set_title("Receiver trace (1 RX) — red lines mark selected times")
ax0.set_xlabel("time [ms]")
ax0.set_ylabel("pressure")

# Bottom row: snapshots with unified norm (vmin/vmax)
extent = img_grid.extent  # (xmin,xmax,ymin,ymax)
for k, idx in enumerate(idx_sel):
    ax = fig.add_subplot(gs[1, k])
    im = ax.imshow(
        WF[idx],
        origin="lower",
        extent=extent,
        cmap="seismic",
        norm=norm,  # <-- unified scaling here
    )
    ax.scatter(
        pair_positions[0, 0],
        pair_positions[1, 0],
        marker="^",
        s=40,
        edgecolors="k",
        label="TX",
    )
    ax.scatter(
        pair_positions[0, 1],
        pair_positions[1, 1],
        marker="s",
        s=40,
        edgecolors="k",
        label="RX",
    )
    if k == 0:
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title(f"t = {t[idx] * 1e3:.2f} ms")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

# Shared colorbar (uses same norm)
cax = fig.add_axes([0.92, 0.12, 0.015, 0.58])
plt.colorbar(im, cax=cax, label="pressure")

plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.97])
plt.show()
