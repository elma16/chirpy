import h5py
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import time
from functions.HelmholtzSolver import HelmholtzSolver
from functions.ringingRemovalFilt import ringingRemovalFilt
from scipy.io import savemat
import os

# set the filename and file path
filename = 'kWave_BreastCT'
# file_path = f'SampleData/{filename}.mat'
file_path = os.path.abspath(f'SampleData/{filename}.mat')

# Load the .mat file
with h5py.File(file_path, 'r') as f:
    data = {}

    # get the keys of the file
    keys = list(f.keys())

    # Load the data into a dictionary
    for key in tqdm(keys, desc="Loading .mat file", unit="var"):
        arr = np.array(f[key])
        if arr.ndim > 1:
            arr = arr.T
        data[key] = np.asfortranarray(arr)  # Convert to Fortran order
        print(f"Loaded {key} with shape {data[key].shape}.")

print("Loading complete.")

xi_orig = data.get('xi_orig')
yi_orig = data.get('yi_orig')
C = data.get('C')
atten = data.get('atten')
time_data = data.get('time')
transducerPositionsXY = data.get('transducerPositionsXY')
full_dataset = data.get('full_dataset')

# get the number of elements
numElements = transducerPositionsXY.shape[1]
print(f"Number of transducers: {numElements}")

# Convert full_dataset to single-precision floating-point numbers.
fsa_rf_data = full_dataset.astype(np.float64, copy=False)
del full_dataset

# Extract the Desired Frequency for Waveform Inversion
fDATA_SoS = np.arange(0.3, 1.3, 0.05).reshape(1, -1) * 1e6  # SoS-only Iteration frequency [Hz] (0.3-1.25 MHz)
fDATA_SoSAtten = np.arange(0.325, 1.325, 0.05).reshape(1,
                                                       -1) * 1e6  # SoS/Attenuation Iteration Frequency [Hz] (0.325-1.375 MHz)
fDATA = np.concatenate([fDATA_SoS, fDATA_SoSAtten], axis=1)  # Combination of all frequencies [Hz]

# Attenuation Iterations Always Happen After SoS Iterations
niterSoSPerFreq = np.concatenate([np.full_like(fDATA_SoS, 3), np.full_like(fDATA_SoSAtten, 3)],
                                 axis=1)  # The number of iterations of sound velocity at each frequency
niterAttenPerFreq = np.concatenate([np.zeros_like(fDATA_SoS), np.full_like(fDATA_SoSAtten, 3)],
                                   axis=1)  # The number of attenuation iterations for each frequency

# Discrete-Time Fourier Transform (DTFT) - Not an FFT though!
dt = np.diff(time_data)
dt_mean = np.mean(dt) if dt.size > 0 else np.nan  # matlab will return nan if dt is empty
DTFT = np.exp(-1j * 2 * np.pi * np.outer(fDATA, time_data)) * dt_mean

# Geometric TOF Based on Sound Speed
c_geom = 1540  # Sound Speed [mm/us]
# Extract the transducer positions
transducerPositionsX = transducerPositionsXY[0, :]
transducerPositionsY = transducerPositionsXY[1, :]
# Initialize the Geometric Time-of-Flight (TOF) Matrix
geomTOFs = np.zeros((numElements, numElements))  # N*N

# Geometric TOF
for col in range(numElements):
    geomTOFs[:, col] = np.sqrt(
        (transducerPositionsX - transducerPositionsX[col]) ** 2 +
        (transducerPositionsY - transducerPositionsY[col]) ** 2
    ) / c_geom
print(f"geomTOFs: {geomTOFs.shape}")

del col, transducerPositionsXY

# Window Time Traces - Extract the Frequencies for Waveform Inversion
REC_DATA = np.zeros((numElements, numElements, fDATA.shape[1]), dtype=np.complex128)
sos_perc_change_pre = 0.05
sos_perc_change_post = np.inf  # MATLAB `Inf` 对应 Python `np.inf`
# Define the Twin Peaks
twinpre = sos_perc_change_pre * np.max(geomTOFs)
twinpost = sos_perc_change_post * np.max(geomTOFs)


# Define the Subplus Function
def subplus(x):
    return np.maximum(x, 0)


# Loop through each element
for tx_element in range(numElements):
    times_tx = geomTOFs[tx_element, :]  # Extract the tof for the current element
    TIMES_TX, TIME = np.meshgrid(times_tx, time_data, indexing='ij')
    window = np.exp(-0.5 * (subplus(TIME - TIMES_TX) / twinpost +
                            subplus(TIMES_TX - TIME) / twinpre) ** 2)
    window = window.T

    REC_DATA[tx_element, :, :] = (DTFT @ (window * fsa_rf_data[:, :, tx_element])).T

############################################################################################################
# Waveform Inversion

# Create Sound Speed Map and Transducer Ring
# dxi = 0.3e-3
dxi = 0.6e-3  # lower resolution considering for limit computing resource
xmax = 120e-3
# xi = np.arange(-xmax, xmax + dxi, dxi).reshape(1, -1)  # include xmax
npts = int(round((xmax - (-xmax)) / dxi)) + 1  # = 401
xi = np.linspace(-xmax, xmax, npts).reshape(1, -1)
yi = xi
Nxi = xi.size
Nyi = yi.size

# Create a meshgrid
Xi, Yi = np.meshgrid(xi, yi, indexing='xy') 

# ring array transducers
x_circ = transducerPositionsX.reshape(1, -1)
y_circ = transducerPositionsY.reshape(1, -1)

# Create KDTree for nearest neighbor search
xi_tree = cKDTree(xi.reshape(-1, 1))
yi_tree = cKDTree(yi.reshape(-1, 1))

x_idx = xi_tree.query(x_circ.reshape(-1, 1))[1]  # the nearest neighbor x index
y_idx = yi_tree.query(y_circ.reshape(-1, 1))[1]  # the nearest neighbor y index

# calculate the linear index
ind = np.ravel_multi_index((y_idx, x_idx), (Nyi, Nxi), order='F')

# Create Mask Matrix
msk = np.zeros((Nyi, Nxi), dtype=int)
msk.flat[ind] = 1  # set 1 at the transducer positions

# Parameters
h = dxi  # [m]
g = 1  # (grid spacing in Y)/(grid spacing in X)
alphaDB = 0.0  # Attenuation [dB/(MHz*mm)]
alphaNp = (np.log(10) / 20) * alphaDB * ((1e3) / (1e6))  # Attenuation [Np/(Hz*m)]

# Make Spatially Varying Attenuation [Np/(Hz m)]
ATTEN = np.full((Nyi, Nxi), alphaNp)

# Solve Options for Helmholtz Equation
sign_conv = -1  # Sign Convention
a0 = 10.0  # PML Constant
L_PML = 9.0e-3  # Thickness of PML


# Conversion of Units for Attenuation Map
Np2dB = 20 / np.log(10)
slow2atten = (1e6) / (1e2)  # Hz to MHz; m to cm

# Compute Phase Screen to Correct for Discretization of Element Positions
# Times for Discretized Positions Assuming Constant Sound Speed
# Calculate the coordinates corresponding to the discretized positions.
x_circ_disc = xi.flatten()[x_idx].reshape(1, -1)
y_circ_disc = yi.flatten()[y_idx].reshape(1, -1)

# Calculate the grid of the transmitting and receiving sensors.
x_circ_disc_rx, x_circ_disc_tx = np.meshgrid(x_circ_disc, x_circ_disc, indexing='ij')
y_circ_disc_rx, y_circ_disc_tx = np.meshgrid(y_circ_disc, y_circ_disc, indexing='ij')
# Calculate the geometric time-of-flight (TOF) for the discretized positions.
geomTOFs_disc = np.sqrt((x_circ_disc_tx - x_circ_disc_rx) ** 2 +
                        (y_circ_disc_tx - y_circ_disc_rx) ** 2) / c_geom
# Time-of-Flight Error for Discretized Positions
geomTOFs_error = geomTOFs_disc - geomTOFs

# Phase Screen and Gain Correction
PS = np.zeros((numElements, numElements, fDATA.shape[1]), dtype=np.complex128)
for f_idx in range(fDATA.shape[1]):
    PS[:, :, f_idx] = np.exp(1j * sign_conv * 2 * np.pi * fDATA[0, f_idx] * geomTOFs_error)

REC_DATA = REC_DATA.astype(np.complex128)  # convert to complex
REC_DATA *= PS
del  PS, geomTOFs_error, geomTOFs_disc, geomTOFs

# Which Subset of Transmits to Use
dwnsmp = 1  # can be 1, 2, or 4 (faster with more downsampling)
# NOTE: dwnsmp = 1 to get the results in the paper
tx_include = np.arange(0, numElements, dwnsmp)
# Extract the Desired Transmits
REC_DATA = REC_DATA[tx_include, :, :]
REC_DATA[np.isnan(REC_DATA)] = 0  # Eliminate Blank Channel

# Extract Subset of Times within Acceptance Angle
numElemLeftRightExcl = 63
elemLeftRightExcl = np.arange(-numElemLeftRightExcl, numElemLeftRightExcl + 1).reshape(1, -1)

elemInclude = np.ones((numElements, numElements), dtype=bool)

for tx_element in range(numElements):
    # index that the current transmitter needs to exclude.
    elemLeftRightExclCurrent = elemLeftRightExcl + (tx_element + 1)

    # Handling Out-of-Bounds Indexes
    elemLeftRightExclCurrent[elemLeftRightExclCurrent < 1] += numElements
    elemLeftRightExclCurrent[elemLeftRightExclCurrent > numElements] -= numElements

    elemInclude[tx_element, elemLeftRightExclCurrent - 1] = False

# Remove Outliers from Observed Signals Prior to Waveform Inversion
perc_outliers = 0.99  # Confidence Interval Cutoff
for f_idx in range(fDATA.shape[1]):
    REC_DATA_SINGLE_FREQ = REC_DATA[:, :, f_idx].copy()
    signalMagnitudes = elemInclude[tx_include, :] * np.abs(REC_DATA_SINGLE_FREQ)
    num_outliers = int(np.ceil((1 - perc_outliers) * signalMagnitudes.size))
    idx_outliers = np.argsort(signalMagnitudes.flatten())[-num_outliers:]
    REC_DATA_SINGLE_FREQ.flat[idx_outliers] = 0
    REC_DATA[:, :, f_idx] = REC_DATA_SINGLE_FREQ

# Initial Constant Sound Speed Map [m/s]
c_init = 1480  # Initial Homogeneous Sound Speed [m/s] Guess
VEL_INIT = np.full((Nyi, Nxi), c_init, dtype=np.float64)

# Initial Constant Attenuation [Np/(Hz m)]
ATTEN_INIT = np.full((Nyi, Nxi), 0 * alphaNp, dtype=np.float64)

############################################################################################################
# (Nonlinear) Conjugate Gradient
search_dir = np.zeros((Nyi, Nxi), dtype=np.float64)  # Conjugate Gradient Direction
gradient_img_prev = np.zeros((Nyi, Nxi), dtype=np.float64)  # Previous Gradient Image

# initialize
VEL_ESTIM = VEL_INIT.copy()  # velocity estimate
ATTEN_ESTIM = ATTEN_INIT.copy()  # attenuation estimate

# initial slowness Image [s/m]
SLOW_ESTIM = 1.0 / VEL_ESTIM + 1j * np.sign(sign_conv) * ATTEN_ESTIM / (2 * np.pi)

# Parameters for Ringing Removal Filter
c0 = np.mean(VEL_ESTIM)
cutoff = 0.75
ord = np.inf

# Values to Save at Each Iteration
Niter = int(np.sum(niterSoSPerFreq) + np.sum(niterAttenPerFreq))
VEL_ESTIM_ITER = np.zeros((Nyi, Nxi, Niter), dtype=np.float64)
ATTEN_ESTIM_ITER = np.zeros((Nyi, Nxi, Niter), dtype=np.float64)
GRAD_IMG_ITER = np.zeros((Nyi, Nxi, Niter), dtype=np.float64)
SEARCH_DIR_ITER = np.zeros((Nyi, Nxi, Niter), dtype=np.float64)


num_freqs = len(fDATA.squeeze())
# num_freqs = 1

num_sources_total = len(tx_include)

total_iter_so_far = 0  



# initial plot
vmin_true, vmax_true = float(C.min()), float(C.max())
attenrange = 10 * np.array([-1, 1])

fig, axes = plt.subplots(2, 3, figsize=(9, 6))

im1 = axes[0, 0].imshow(np.zeros((len(yi), len(xi))), cmap='gray', aspect='auto')
axes[0, 0].set_title('Estimated Wave Velocity')

# True Wave Velocity 
im3 = axes[0, 1].imshow(C, vmin=vmin_true, vmax=vmax_true, cmap='gray', aspect='auto')
axes[0, 1].set_title('True Wave Velocity')

# Search Direction Iteration
im5 = axes[0, 2].imshow(np.zeros((len(yi), len(xi))), cmap='gray', aspect='auto')
axes[0, 2].set_title('Search Direction Iteration')

# Estimated Attenuation
im2 = axes[1, 0].imshow(np.zeros((len(yi), len(xi))),
                        vmin=attenrange[0], vmax=attenrange[1],
                        cmap='gray', aspect='auto')
axes[1, 0].set_title('Estimated Attenuation')

# true Attenuation
im4 = axes[1, 1].imshow(atten, vmin=attenrange[0], vmax=attenrange[1],
                        cmap='gray', aspect='auto')
axes[1, 1].set_title('Attenuation')

# Gradient Iteration
im6 = axes[1, 2].imshow(np.zeros((len(yi), len(xi))), cmap='gray', aspect='auto')
axes[1, 2].set_title('Gradient Iteration')

plt.tight_layout()
plt.pause(0.01)



for f_idx in range(num_freqs):  # f_idx in [0..num_freqs-1]
    # print('>'*50, niterSoSPerFreq[f_idx])
    this_freq = fDATA.squeeze()[f_idx]
    nIter_sos = int(niterSoSPerFreq.squeeze()[f_idx])
    nIter_att = int(niterAttenPerFreq.squeeze()[f_idx])
    nIter_total = nIter_sos + nIter_att
    # nIter_total = 1
    

    for iter_f_idx in range(1, nIter_total+1):  # 1..nIter_total
        tic = time.time()

        iter_global = iter_f_idx + total_iter_so_far
        
        # Step 0: Reset CG at Each Frequency (SoS and Attenuation)
        if ((iter_f_idx == 1) or (iter_f_idx == 1 + nIter_sos)):
            search_dir = np.zeros((Nyi, Nxi), dtype=np.float64)
            gradient_img_prev = np.zeros((Nyi, Nxi), dtype=np.float64)

        # Attenuation Iterations Happen After SoS Iterations
        if iter_f_idx > nIter_sos:
            updateAttenuation = True
            print("updateAttenuation")
        else:
            updateAttenuation = False

        gradient_img = np.zeros((Nyi, Nxi), dtype=np.float64)

        # Step 1: Accumulate Backprojection Over Each Element
        # Generate Sources (SRC)
        SRC = np.zeros((Nyi, Nxi, num_sources_total))
        for e_idx in range(num_sources_total):
            xx = x_idx[tx_include[e_idx]]
            yy = y_idx[tx_include[e_idx]]
            SRC[yy, xx, e_idx] = 1.0  # single element source

        # Forward Solve Helmholtz Equation
        HS = HelmholtzSolver(xi, yi, VEL_ESTIM, ATTEN_ESTIM, this_freq, sign_conv, a0, L_PML, canUseGPU=False)
        WVFIELD, VIRT_SRC = HS.solve(SRC, adjoint=False)

        # Virtual Sources for Attenuation
        if updateAttenuation:
            # VIRT_SRC = 1i*sign(sign_conv)*VIRT_SRC
            # => 1j * np.sign(sign_conv) * VIRT_SRC
            VIRT_SRC = 1j * np.sign(sign_conv) * VIRT_SRC

        # Build Adjoint Sources
        scaling = np.zeros((num_sources_total,), dtype=np.complex128)
        ADJ_SRC = np.zeros((Nyi, Nxi, num_sources_total), dtype=np.complex128)

        for e_idx in range(num_sources_total):
            WVFIELD_elmt = WVFIELD[:,:,e_idx]

            sim_indices = elemInclude[tx_include[e_idx], :] 
            REC_SIM = WVFIELD_elmt.ravel(order='F')[ind[sim_indices]]  
            # true measurement:
            REC = REC_DATA[e_idx, sim_indices, f_idx]  
            # Source scaling
            # (REC_SIM(:)'*REC(:)) / (REC_SIM(:)'*REC_SIM(:))

            top = np.dot(np.conj(REC_SIM), REC)
            bot = np.dot(np.conj(REC_SIM), REC_SIM)

            scaling[e_idx] = top / bot if bot != 0 else 0

            # ADJ_SRC_elmt
            ADJ_SRC_elmt = np.zeros((Nyi, Nxi), dtype=np.complex128)
            # ADJ_SRC_elmt(ind(...)) = scaling(e_idx)*REC_SIM - REC
            ADJ_vals = scaling[e_idx]*REC_SIM - REC
            # ADJ_SRC_elmt.flat[ind[sim_indices]] = ADJ_vals
            ys, xs = np.unravel_index(ind[sim_indices], (Nyi, Nxi), order='F')
            ADJ_SRC_elmt[ys, xs] = ADJ_vals
            ADJ_SRC[:,:, e_idx] = ADJ_SRC_elmt

        # Backproject Error
        ADJ_WVFIELD,_ = HS.solve(ADJ_SRC, adjoint=True)  # ADJ_WVFIELD

        # SCALING for repmat
        SCALING = np.tile(scaling.reshape((1,1,-1)),(Nyi,Nxi,1))
        # SCALING.*VIRT_SRC shape => (Nyi,Nxi,num_sources_total)
        # conj(...) => np.conj(...)
        scaled_VSRC = SCALING * VIRT_SRC
        BACKPROJ_cplx = np.conj(scaled_VSRC) * ADJ_WVFIELD
        BACKPROJ = -np.real(BACKPROJ_cplx)

        # Accumulate Gradient Over Each Element
        for e_idx in range(num_sources_total):
            gradient_img += BACKPROJ[:,:, e_idx]


        # Remove Ringing from Gradient Image
        gradient_img = ringingRemovalFilt(xi, yi, gradient_img, c0, this_freq, cutoff, ord)

        # Step 2: Compute Conjugate Gradient Direction
        if ((iter_f_idx == 1) or (iter_f_idx == 1 + nIter_sos)):
            beta = 0
        else:
            # betaPR = (g*(g-g_prev)) / (g_prev*g_prev)
            # betaFR = (g*g)/(g_prev*g_prev)
            g  = gradient_img.ravel()
            gp = gradient_img_prev.ravel()
            topPR = np.dot(g, (g - gp))
            botPR = np.dot(gp, gp)
            betaPR = topPR / botPR if botPR!=0 else 0

            topFR = np.dot(g, g)
            botFR = botPR  # same as above
            betaFR = topFR / botFR if botFR!=0 else 0

            beta = min( max(betaPR, 0), betaFR )

        # search_dir = beta*search_dir - gradient_img
        search_dir = beta*search_dir - gradient_img
        gradient_img_prev = gradient_img.copy()

        # Step 3: Forward Projection of Current Search Direction
        # PERTURBED_WVFIELD = HS.solve(VIRT_SRC.*search_dir, false);
        #   => wavefield from "SRC = VIRT_SRC * search_dir"
        #   VIRT_SRC shape= (Nyi,Nxi,num_sources_total)
        #   search_dir shape= (Nyi,Nxi) broadcast
        #   => broadcast: (Nyi,Nxi,1)
        search_dir_3D = np.expand_dims(search_dir, axis=2)  # shape(Nyi,Nxi,1)
        new_SRC = VIRT_SRC * search_dir_3D  # elementwise multiply
        PERTURBED_WVFIELD,_ = HS.solve(new_SRC, adjoint=False)

        dREC_SIM = np.zeros((num_sources_total, numElements),     
                    dtype=np.complex128,
                    order='F')                               

        for e_idx in range(len(tx_include)):
            # 1) Take a 2D slice of a certain emission element e_idx
            PERTURBED_WVFIELD_elmt = PERTURBED_WVFIELD[:,:, e_idx]  # shape=(Nyi,Nxi)

            # 2) To access this batch of indices elemInclude(tx_include(e_idx),:)
            row_elem = tx_include[e_idx]                    
            col_indices = elemInclude[row_elem, :]           
            
            sub_1d_indices = ind[ col_indices ]            
            
            # 3)Extract the corresponding position from the wavefield (1D)
            subVals = PERTURBED_WVFIELD_elmt.ravel(order='F')[sub_1d_indices]

            # 4) scaling
            subValsScaled = - (scaling[e_idx] * subVals)

            # 5) Write to row e_idx and column col_indices of dREC_SIM.
            ys, xs = np.unravel_index(ind[col_indices], (Nyi, Nxi), order='F')
            subVals = PERTURBED_WVFIELD_elmt[ys, xs]
            dREC_SIM[e_idx, col_indices] = -scaling[e_idx] * subVals


        # Step 4: Linear Approx. of Exact Line Search
        perc_step_size = 1

        # -------- alpha (line-search step)  --------
        g_flat = gradient_img.ravel(order='F')          
        d_flat = search_dir.ravel(order='F')
        s_flat = dREC_SIM.ravel(order='F')            

        top_alpha = -np.dot(g_flat, d_flat)                  
        bot_alpha = np.sum(np.abs(s_flat)**2, dtype=np.float32) 

        alpha = (top_alpha / bot_alpha) if bot_alpha != 0 else 0.0

        if updateAttenuation:
            # SI = sign(sign_conv)*imag(SLOW_ESTIM) + alpha*search_dir
            # => SLOW_ESTIM = real(SLOW_ESTIM) + i * sign(sign_conv)* SI
            # => wave velocity = 1/ real(SLOW_ESTIM)
            SI = np.sign(sign_conv)*np.imag(SLOW_ESTIM) + perc_step_size*alpha*search_dir
            SLOW_ESTIM = np.real(SLOW_ESTIM) + 1j*(np.sign(sign_conv)*SI)
        else:
            SLOW_ESTIM = SLOW_ESTIM + perc_step_size*alpha*search_dir


        VEL_ESTIM   = 1.0 / np.real(SLOW_ESTIM)
        ATTEN_ESTIM = 2*np.pi*np.imag(SLOW_ESTIM)*np.sign(sign_conv)

        # Save intermediate results
        VEL_ESTIM_ITER[:,:, iter_global-1]   = VEL_ESTIM
        ATTEN_ESTIM_ITER[:,:, iter_global-1] = ATTEN_ESTIM
        GRAD_IMG_ITER[:,:, iter_global-1]    = gradient_img
        SEARCH_DIR_ITER[:,:, iter_global-1]  = search_dir

        
        ####vsualization#######
        # update
        # 1) sound volicity
        im1.set_data(VEL_ESTIM)
        vmin, vmax = float(VEL_ESTIM.min()), float(VEL_ESTIM.max())
        im1.set_clim(vmin, vmax)
        axes[0, 0].set_title(f'Estimated Wave Velocity {total_iter_so_far + iter_f_idx}')

        # 2) search direction
        smin, smax = search_dir.min(), search_dir.max()
        if smin == smax: smin, smax = smin-1, smax+1
        im5.set_data(search_dir)
        im5.set_clim(smin, smax)
        axes[0, 2].set_title(f'Search Direction Iteration {total_iter_so_far + iter_f_idx}')

        # 3) attentuation
        im2.set_data(Np2dB * slow2atten * ATTEN_ESTIM)
        axes[1, 0].set_title(f'Estimated Attenuation {total_iter_so_far + iter_f_idx}')

        # 4) gradient
        gmin, gmax = (-gradient_img).min(), (-gradient_img).max()
        if gmin == gmax: gmin, gmax = gmin-1, gmax+1
        im6.set_data(-gradient_img)
        im6.set_clim(gmin, gmax)
        axes[1, 2].set_title(f'Gradient Iteration {total_iter_so_far + iter_f_idx}')

        fig.canvas.flush_events()
        plt.pause(0.001)

        print(f'Iteration {iter_global}')
        print(f'Elapsed time: {time.time() - tic:.2f} s')


    total_iter_so_far += nIter_total

print('Finished!')
plt.ioff()
plt.show()


# Plot Final Reconstructions
plt.figure()
plt.subplot(2, 2, 1)
vmin, vmax = float(VEL_ESTIM.min()), float(VEL_ESTIM.max())
plt.imshow(VEL_ESTIM, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
           vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
plt.title(f'Estimated Wave Velocity {total_iter_so_far}')

plt.subplot(2, 2, 2)
plt.imshow(Np2dB*slow2atten*ATTEN_ESTIM,
           extent=[xi.min(), xi.max(), yi.max(), yi.min()],
           vmin=attenrange[0], vmax=attenrange[1], cmap='gray', aspect='auto')
plt.title(f'Estimated Attenuation {total_iter_so_far}')

plt.subplot(2, 2, 3)
plt.imshow(C, extent=[xi_orig.min(), xi_orig.max(), yi_orig.max(), yi_orig.min()],
           vmin=vmin_true, vmax=vmax_true, cmap='gray', aspect='auto')
plt.title('True Wave Velocity')

plt.subplot(2, 2, 4)
plt.imshow(atten, extent=[xi_orig.min(), xi_orig.max(), yi_orig.min(), yi_orig.max()],
           vmin=attenrange[0], vmax=attenrange[1], cmap='gray', aspect='auto')
plt.title('Attenuation')

plt.tight_layout()
plt.show()

# Save the Result to File
filename_results = f"Results/{filename}_WaveformInversionResults.mat"
savemat(filename_results, {
    'xi': xi,
    'yi': yi,
    'fDATA': fDATA,
    'niterAttenPerFreq': niterAttenPerFreq,
    'niterSoSPerFreq': niterSoSPerFreq,
    'VEL_ESTIM_ITER': VEL_ESTIM_ITER,
    'ATTEN_ESTIM_ITER': ATTEN_ESTIM_ITER,
    'GRAD_IMG_ITER': GRAD_IMG_ITER,
    'SEARCH_DIR_ITER': SEARCH_DIR_ITER
}, do_compression=True)
print(f"Results saved to {filename_results}")