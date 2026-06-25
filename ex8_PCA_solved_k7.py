import numpy as np
import matplotlib.pyplot as plt

# Section 1 average burst - the average ideal where everything is computed from 
# a corrected baseline and this is a trial by trial filtering and you run SVD and derive
#PCs

#and with individual bursts, real brains don't average things out and 
#and response might be faster in trial 3 by 5 seconds compared to trial 1 
#so for individual bursts we do not run SVD again
#You keep the exact same timeline filters (U) that you calculated from the averaged data, and you press them like a cookie cutter 
# against the raw, messy, single-trial bursts:


# UNDERSTANDING THE SVD MATRIX MULTIPLICATION (U.T @ burst_centered)
# 
# 1. WHAT 'U' ACTUALLY IS:
#    When we run SVD on the averaged matrix (X.T), U is generated with a size of
#    [Nt, Nt] (Time Points x Time Points).
#    
#    Each column of U is a literal weight profile across time for that PC.
#    For example, for 5 time points, the first column (PC1) might look like:
#    U[:, 0] = [0.1, 0.4, 0.8, 0.4, 0.1]
#    
#    This vector says: "To capture PC1, ignore the start and end of the trial,
#    and pay maximum attention to the middle (index 2, where weight is 0.8)."
#
# 2. THE MATHEMATICAL PROJECTION:
#    When we calculate 'U.T @ burst_centered', we take this time-template and
#    multiply it directly against the raw burst spike rates, point-by-point.
#
#    EXAMPLE - BURST 1 (Fired perfectly on time):
#    Raw data over 5 time points = [12, 45, 90, 30, 10]
#    Dot product = (0.1*12) + (0.4*45) + (0.8*90) + (0.4*30) + (0.1*10)
#                = 1.2      + 18       + 72       + 12       + 1
#                = 104.2  <-- High score because it peaked at index 2
#
#    EXAMPLE - BURST 2 (Fired late / mismatched):
#    Raw data over 5 time points = [10, 15, 12, 40, 90]
#    Dot product = (0.1*10) + (0.4*15) + (0.8*12) + (0.4*40) + (0.1*90)
#                = 1        + 6        + 9.6      + 16       + 9
#                = 41.6   -  Low score because it missed the template focus
#
# 3. WHAT THE INDIVIDUAL BURST PLOT SHOWS:
#    By calculating this for every burst and superimposing them on one plot:
#    - If lines pack tightly together: The brain fires identically every trial.
#    - If lines shift or flatten: You are seeing real-time timing jitters and 
#      trial-by-trial variance in the living circuit.

# 1. Load the raw file 
MUA_b_t_g = np.load("/home/keerthie/Desktop/TNS_2/Tutorial_Exercises_Solved/ex8_PCA/MUA_b_t_g.npy", allow_pickle=True)

Nb, Nt, Ng = MUA_b_t_g.shape   # 10, 117, 20
ti = np.arange(Nt)              # simple time axis 0 to 116 since no ti provided
# 4. Average over bursts (axis 0) to get (Nt, Ng), then transpose to (Ng, Nt)
MUA_g_t = np.mean(MUA_b_t_g, axis=0).T   



# Plot each neuron group's activity over time
plt.figure(figsize=(8, 4))
for g in range(Ng):
    plt.plot(ti, MUA_g_t[g, :], color='#5b7c99', alpha=0.5)
plt.xlabel('time [ms]')
plt.ylabel('activity [Hz/neuron]')
plt.title('mean activity per neuron group')
plt.tight_layout()
plt.show()

# Subtract the row mean from each row
# keepdims=True keeps shape (Ng, 1) so subtraction broadcasts correctly across columns
row_means = MUA_g_t.mean(axis=1, keepdims=True)  # shape (Ng, 1)
X0 = MUA_g_t - row_means                          # shape (Ng, Nt), each row now sums to zero

# Compute covariance matrix
# X0 @ X0.T gives all pairwise dot products between neuron group rows
# dividing by Nt-1 gives the sample covariance
CX = (X0 @ X0.T) / (Nt - 1)   # shape (Ng, Ng)

# Plot covariance matrix
plt.figure(figsize=(5, 4))
plt.pcolor(CX, cmap='hot')
plt.colorbar()
plt.xlabel('neuron group')
plt.ylabel('neuron group')
plt.title('covariance matrix C_X')
plt.tight_layout()
plt.show()

# Singular value decomposition
# input must be X0 transposed: shape (Nt, Ng)
# full_matrices=False gives the skinny SVD (avoids unnecessarily large matrices)
# U: shape (Nt, Ng), S: shape (Ng,) diagonal values only, Vt: shape (Ng, Ng)
U, S, Vt = np.linalg.svd(X0.T, full_matrices=False)

# Plot singular values to see how many PCs capture significant variance
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(S) + 1), S, color='#5b7c99')
plt.xlabel('principal component')
plt.ylabel('singular value')
plt.title('variance captured by each PC')
plt.tight_layout()
plt.show()

# SECTION 2: TRANSFORMED ACTIVITY


# P = Vt (rows of Vt are the principal components)
# Transform X0 into PC space
# Y has shape (Ng, Nt): rows are now PCs instead of neuron groups
P = Vt
Y = P @ X0   # shape (Ng, Nt)

# Plot each PC's activity over time
plt.figure(figsize=(8, 4))
for g in range(Ng):
    plt.plot(ti, Y[g, :], color='#a8763e', alpha=0.5)
plt.xlabel('time [ms]')
plt.ylabel('activity in PC space')
plt.title('transformed activity Y')
plt.tight_layout()
plt.show()

# Compute covariance of Y and verify it is diagonal
CY = (Y @ Y.T) / (Nt - 1)   # shape (Ng, Ng), should be diagonal

plt.figure(figsize=(5, 4))
plt.pcolor(CY, cmap='hot')
plt.colorbar()
plt.xlabel('PC')
plt.ylabel('PC')
plt.title('covariance matrix C_Y (should be diagonal)')
plt.tight_layout()
plt.show()

# Plot only the first three PCs over time
plt.figure(figsize=(8, 4))
colors = ['#5b7c99', '#a8763e', '#6b8f71']
for i in range(3):
    plt.plot(ti, Y[i, :], color=colors[i], label=f'PC {i+1}')
plt.xlabel('time [ms]')
plt.ylabel('activity')
plt.title('first three principal components')
plt.legend()
plt.tight_layout()
plt.show()

# Denoising: zero all PCs beyond the first three
Y_denoised = Y.copy()
Y_denoised[3:, :] = 0   # keep rows 0, 1, 2 only

# Back project into neuron group space
# P.T is the inverse of P because P is orthonormal
X0_denoised = P.T @ Y_denoised      # shape (Ng, Nt)
X_denoised = X0_denoised + row_means  # add back the row means we subtracted earlier

# Plot denoised activity per neuron group
plt.figure(figsize=(8, 4))
for g in range(Ng):
    plt.plot(ti, X_denoised[g, :], color='#6b8f71', alpha=0.5)
plt.xlabel('time [ms]')
plt.ylabel('activity [Hz/neuron]')
plt.title('denoised activity (first 3 PCs only)')
plt.tight_layout()
plt.show()


# SECTION 3: INDIVIDUAL BURSTS


# Reshape MUA_b_t_g to work with individual bursts
# MUA_b_t_g shape: (Nb, Nt, Ng)
# We want shape (Ng, Nt) for each burst, then project into PC space

plt.figure(figsize=(7, 6))
ax = plt.axes(projection='3d')

for b in range(Nb):
    # get one burst: shape (Nt, Ng), then transpose to (Ng, Nt)
    burst = MUA_b_t_g[b, :, :].T          # shape (Ng, Nt)

    # subtract the same row means from the average burst
    # this puts individual bursts in the same zero-mean coordinate system
    burst_0 = burst - row_means            # shape (Ng, Nt)

    # project into PC space using P found from the average burst
    Y_burst = P @ burst_0                  # shape (Ng, Nt)

    # plot trajectory through PC1, PC2, PC3 space
    ax.plot(Y_burst[0, :], Y_burst[1, :], Y_burst[2, :], alpha=0.4, color='#5b7c99')
    # mark the starting point of each burst
    ax.scatter(Y_burst[0, 0], Y_burst[1, 0], Y_burst[2, 0], color='#a8763e', s=10)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('individual burst trajectories in PC space')
plt.tight_layout()
plt.show()