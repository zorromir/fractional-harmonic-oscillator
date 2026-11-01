# -*- coding: utf-8 -*-
"""
Created on Mon May 26 09:20:14 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Parameters
T = 40
N = 2000
dt = T / N
t = np.linspace(0, T, N + 1)
f = np.cos(t)
a = 1.0
b = 1.0

# L2 weights
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

# L1 weights
def l1_weights(beta, n, dt):
    b_weights = np.zeros(n)
    for k in range(n):
        b_weights[k] = ((k + 1)**(1 - beta) - k**(1 - beta))
    return (dt**(-beta) / gamma(2 - beta)) * b_weights

# Ranges
alpha_vals = np.linspace(1.2, 1.8, 2)
beta_vals = np.linspace(0.2, 0.8, 2)
X_alpha = np.zeros((N + 1, len(alpha_vals)))
X_beta = np.zeros((N + 1, len(beta_vals)))

# Fixed beta
fixed_beta = 0.6
for j, alpha in enumerate(alpha_vals):
    x = np.zeros(N + 1)
    x[0] = x[1] = 1.0
    for n in range(1, N):
        W = l2_weights(alpha, n, dt)
        b_w = l1_weights(fixed_beta, n, dt)
        sum_l2 = sum(W[k + 1] * x[n - k] for k in range(n + 1))
        sum_l1 = sum(b_w[n - k - 1] * (x[k + 1] - x[k]) for k in range(n))
        x[n + 1] = (f[n] - sum_l2 - a * sum_l1 - b * x[n]) / W[0]
    X_alpha[:, j] = x

# Fixed alpha
fixed_alpha = 1.8
for j, beta in enumerate(beta_vals):
    x = np.zeros(N + 1)
    x[0] = x[1] = 1.0
    for n in range(1, N):
        W = l2_weights(fixed_alpha, n, dt)
        b_w = l1_weights(beta, n, dt)
        sum_l2 = sum(W[k + 1] * x[n - k] for k in range(n + 1))
        sum_l1 = sum(b_w[n - k - 1] * (x[k + 1] - x[k]) for k in range(n))
        x[n + 1] = (f[n] - sum_l2 - a * sum_l1 - b * x[n]) / W[0]
    X_beta[:, j] = x

# Trim early values
cut_idx = 5
t_cut = t[cut_idx:]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Alpha variation plot
for j, alpha in enumerate(alpha_vals):
    axs[0].plot(t_cut, X_alpha[cut_idx:, j], label=fr'$\alpha = {alpha:.2f}$')

axs[0].set_xlabel(r'$t$')
axs[0].set_ylabel(r'$x(t)$')
axs[0].legend()

# Beta variation plot
for j, beta in enumerate(beta_vals):
    axs[1].plot(t_cut, X_beta[cut_idx:, j], label=fr'$\beta = {beta:.2f}$')

axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$x(t)$')
axs[1].legend()

plt.tight_layout()
plt.show()
