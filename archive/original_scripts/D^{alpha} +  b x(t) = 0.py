# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:14:27 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Parameters
T = 20
N = 100
dt = T / N
t = np.linspace(0, T, N + 1)
omega = 1.0
x0 = 1.0
x1 = 0.0

# L2 weights function
def l2_weights(alpha, n, dt):
    W = np.zeros(n + 2)
    for k in range(-1, n + 1):
        idx = k + 1
        if k == -1:
            W[idx] = 1
        elif k == 0:
            W[idx] = 2**(2 - alpha) - 3
        elif 1 <= k <= n - 2:
            W[idx] = (k + 2)**(2 - alpha) - 3 * (k + 1)**(2 - alpha) + 3 * k**(2 - alpha) - (k - 1)**(2 - alpha)
        elif k == n - 1:
            W[idx] = -2 * n**(2 - alpha) + 3 * (n - 1)**(2 - alpha) - (n - 2)**(2 - alpha)
        elif k == n:
            W[idx] = n**(2 - alpha) - (n - 1)**(2 - alpha)
    return W * dt**(-alpha) / gamma(3 - alpha)

# L2 solver
def solve_fde(alpha):
    x = np.zeros(N + 1)
    x[0] = x0
    x[1] = x0 + dt * x1
    for n in range(1, N):
        W = l2_weights(alpha, n, dt)
        sum_l2 = sum(W[k + 1] * x[n - k] for k in range(n + 1))
        x[n + 1] = (-omega**2 * x[n] - sum_l2) / W[0]
    return x

# Run for different alpha values
alphas = [1.4, 1.8]
results = [(alpha, solve_fde(alpha)) for alpha in alphas]
x_classic = np.cos(omega * t)  # klassieke oscillator

# Plot time-domain solutions
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

for i, (alpha, x) in enumerate(results):
    axs[i].plot(t, x, label=rf"Fractionele oscillator ($\alpha = {alpha}$)", linewidth=3, color='black')
    axs[i].plot(t, x_classic, label="Klassieke oscillator", linestyle='--', color='gray', linewidth=2)
    axs[i].set_title(rf"Oplossing $x(t)$ voor $\alpha = {alpha}$", fontsize=16)
    axs[i].set_xlabel("Tijd $t$")
    axs[i].set_ylabel("Verplaatsing $x(t)$")
    axs[i].grid(True, linestyle='--', linewidth=0.7)
    axs[i].legend()

plt.tight_layout()
plt.show()
