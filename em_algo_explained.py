import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ---------------------------------------------------------
# Load the data
# u_obs ends up as a flat 1D array of inter-spike intervals in ms
# e.g. [312, 367, 289, 401, ...] with 10000 values
# ---------------------------------------------------------
data2 = np.load("/home/keerthie/Desktop/TNS_2/Tutorial_Exercises_Solved/ex7_expectation_maximisation/observations.npy", allow_pickle=True).item()["samples"].tolist()
u_obs = np.array(data2).flatten()

# ---------------------------------------------------------
# Hand-fit starting point (Task 1)
# These are your by-eye guesses from looking at the histogram
# K     = how many bumps you counted
# gammas = what fraction of the data each cause contributes (must sum to 1)
# lambdas = where each bump is centered on the x-axis (in ms)
# ---------------------------------------------------------
K = 2
gammas = np.array([0.25, 0.75])
lambdas = np.array([300, 365])

# ---------------------------------------------------------
# E STEP: for every observation, compute how likely each cause produced it
#
# Output P has shape (K, N):
#   - each row k is "P(cause k | every observation)" across all N observations
#   - each column n is "the K probabilities for observation n" which sum to 1
#
# How it works:
#   raw[k, :] = gamma_k * p(u | lambda_k)    <- numerator of Bayes rule for cause k
#               (how common cause k is) * (how well cause k explains each u)
#   P = raw / sum of raw across all causes    <- divide by denominator to get proper probabilities
# ---------------------------------------------------------
def Expect(u_obs, gammas, lambdas):
    K = len(gammas)
    N = len(u_obs)

    # preallocate empty container: K rows, N columns
    raw = np.zeros((K, N))

    for k in range(K):
        # for cause k: multiply its fraction (gamma) by the Poisson probability
        # of each observation given that cause's rate (lambda)
        # poisson.pmf is the same formula lambda^u * e^(-lambda) / u!
        # but computed in a numerically stable way (avoids factorial blowing up)
        raw[k, :] = gammas[k] * poisson.pmf(u_obs, lambdas[k])

    # divide each raw score by the total raw score across all causes
    # this rescales so that for every observation, the K probabilities sum to 1
    # axis=0 means sum down the rows (across causes) for each column (observation)
    # keepdims=True keeps shape (1, N) so the division lines up correctly against (K, N)
    P = raw / raw.sum(axis=0, keepdims=True)

    return P

# ---------------------------------------------------------
# M STEP: use the soft assignments from E step to recompute gamma and lambda
#
# gamma_k  = average of P(k|u) across all observations
#           = how much of the data belongs to cause k on average
#
# lambda_k = weighted average of u, weighted by P(k|u)
#           = what is the typical gap length for cause k specifically
#           numerator:   sum of (each observation * its probability of belonging to k)
#           denominator: total weight belonging to k (= gamma_k * N, or equivalently P.sum(axis=1))
# ---------------------------------------------------------
def Maximise(u_obs, P):
    # average across all observations (axis=1 = across columns) for each cause row
    gammas_new = P.mean(axis=1)

    # P * u_obs: numpy lines up the N-length u_obs against each K row of P automatically
    # .sum(axis=1): sum across observations, giving weighted numerator for each cause
    # / P.sum(axis=1): divide by total weight for each cause to get a proper average
    lambdas_new = (P * u_obs).sum(axis=1) / P.sum(axis=1)

    return gammas_new, lambdas_new

# ---------------------------------------------------------
# MAIN LOOP: alternate E and M up to 200 times
#
# Each iteration:
#   E step -> soft assignments P from current gammas and lambdas
#   M step -> updated gammas and lambdas from P
#
# Convergence check: if the largest change in any lambda is smaller
# than 0.01ms, the estimates have stopped moving meaningfully, so stop early
# ---------------------------------------------------------
for i in range(200):
    P = Expect(u_obs, gammas, lambdas)
    gammas_new, lambdas_new = Maximise(u_obs, P)

    if np.max(np.abs(lambdas_new - lambdas)) < 0.01:
        print(f"Converged at iteration {i+1}")
        break

    gammas, lambdas = gammas_new, lambdas_new

print("Final gammas:", gammas)
print("Final lambdas:", lambdas)

# ---------------------------------------------------------
# PLOTTING
#
# histogram: raw counts of the data across 40 bins in range 100-500ms
#
# for each cause k, the fitted Poisson curve height at x is:
#   gamma_k * N * bin_width * poisson.pmf(x, lambda_k)
#
#   gamma_k * N   = how many observations effectively belong to cause k
#   * bin_width   = scale from probability (per single value) to frequency (per bin)
#   * poisson.pmf = the shape of cause k's Poisson distribution
#
# sum of all components is plotted as a dashed line
# ---------------------------------------------------------
colors = ['#5b7c99', '#a8763e', '#6b8f71', '#9e6b8f']

bins = 40
bin_range = (100, 500)
bin_width = (bin_range[1] - bin_range[0]) / bins

plt.figure(figsize=(8, 5))
plt.hist(u_obs, bins=bins, range=bin_range, color='#cfcabd', edgecolor='white', label='data')

x = np.arange(bin_range[0], bin_range[1])
total = np.zeros_like(x, dtype=float)

for k in range(K):
    component = gammas[k] * len(u_obs) * bin_width * poisson.pmf(x, lambdas[k])
    plt.plot(x, component, color=colors[k % len(colors)], label=f'cause {k+1}')
    total += component

plt.plot(x, total, color='#333333', linestyle='--', label='sum')
plt.xlabel('inter-spike interval [ms]')
plt.ylabel('frequency')
plt.legend()
plt.show()