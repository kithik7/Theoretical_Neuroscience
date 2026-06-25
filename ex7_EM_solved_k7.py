
"""
data2 = np.load("/home/keerthie/Desktop/TNS_2/Tutorial_Exercises_Solved/ex7_expectation_maximisation/observations.npy", allow_pickle=True).item()["samples"].tolist()
plt.hist(data2, bins=40, range=(100,500))
plt.xlabel("inter-spike interval [ms]")
plt.ylabel("frequency")
plt.show()

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import poisson

#load observatons npy first 
data2 = np.load("/home/keerthie/Desktop/TNS_2/Tutorial_Exercises_Solved/ex7_expectation_maximisation/observations.npy", allow_pickle=True).item()["samples"].tolist()
u_obs = np.array(data2).flatten()

# Hand-fit starting point (Task 1) - replace these with your own numbers

K = 2    # number of causes you identified by eye
gammas = np.array([0.25, 0.75])
lambdas = np.array([300, 365])

""" 
Step 1 : Expectation: we assign soft probabilities to each cause from P(u|a) , 
apply bayes and get p(a|u) for each cause and observation. 
we get the soft assignment of each observation to each cause. We can then use these soft assignments to recompute 
the parameters of the model in the M step.
"""
def Expect(u_obs, gammas, lambdas):
    K = len(gammas)
    N = len(u_obs)
    raw = np.zeros((K, N))
    for k in range(K):
        raw[k, :] = gammas[k] * poisson.pmf(u_obs, lambdas[k])
    P = raw / raw.sum(axis=0, keepdims=True)
    return P


def Maximise(u_obs, P):
    gammas_new = P.mean(axis=1)
    lambdas_new = (P * u_obs).sum(axis=1) / P.sum(axis=1)
    return gammas_new, lambdas_new

#EM loop for 200 iterations
for i in range(200):
    P = Expect(u_obs, gammas, lambdas)
    gammas_new, lambdas_new = Maximise(u_obs, P)
    
    if np.max(np.abs(lambdas_new - lambdas)) < 0.01:
        print(f"Converged at iteration {i+1}")
        break
    
    gammas, lambdas = gammas_new, lambdas_new

"""
after each M step, check how much the lambdas changed from the previous iteration. 
If the biggest change across all causes is smaller than 0.01ms, the estimates have effectively stopped moving, 
so there is no point continuing. Break out of the loop early and print which iteration it converged at.
If it never converges within 200 iterations it just runs all the way through without breaking, which is fine.
"""


#plot histogram of data and the fitted model components and their sum.

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