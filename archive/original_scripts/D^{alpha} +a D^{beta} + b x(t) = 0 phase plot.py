# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:36:51 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Parameters
T = 30
N = 2000
dt = T / N
t = np.linspace(0, T, N + 1)
a = b = 1.0

def l2_weights(alpha, n, dt):
    W = np.zeros(n + 2)
    for k in range(-1, n + 1):
        idx = k + 1
        if k == -1: W[idx] = 1
        elif k == 0: W[idx] = 2**(2 - alpha) - 3
        elif 1 <= k <= n - 2:
            W[idx] = (k + 2)**(2 - alpha) - 3 * (k + 1)**(2 - alpha) + 3 * k**(2 - alpha) - (k - 1)**(2 - alpha)
        elif k == n - 1:
            W[idx] = -2 * n**(2 - alpha) + 3 * (n - 1)**(2 - alpha) - (n - 2)**(2 - alpha)
        elif k == n:
            W[idx] = n**(2 - alpha) - (n - 1)**(2 - alpha)
    return W * dt**(-alpha) / gamma(3 - alpha)

def l1_weights(beta, n, dt):
    return np.array([(k + 1)**(1 - beta) - k**(1 - beta) for k in range(n)]) * dt**(-beta) / gamma(2 - beta)

def solve_fde(alpha, beta):
    x = np.zeros(N + 1)
    x[0] = x[1] = 1.0
    for n in range(1, N):
        W = l2_weights(alpha, n, dt)
        b_w = l1_weights(beta, n, dt)
        sum_l2 = sum(W[k + 1] * x[n - k] for k in range(n + 1))
        sum_l1 = sum(b_w[n - k - 1] * (x[k + 1] - x[k]) for k in range(n))
        x[n + 1] = (- sum_l2 - a * sum_l1 - b * x[n]) / W[0]
    return x

# Solve for two configurations
x1 = solve_fde(1.8, 0.6)
x2 = solve_fde(1.2, 0.2)
dx1 = np.gradient(x1, dt)
dx2 = np.gradient(x2, dt)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(x1, dx1)

axs[0].set_xlabel(r'$x(t)$'); axs[0].set_ylabel(r'$\dot{x}(t)$')
axs[0].set_ylim(-1.5, 1.5)
axs[0].set_xlim(-1.1, 0.99)
axs[0].grid(True)

axs[1].plot(x2, dx2)

axs[1].set_xlabel(r'$x(t)$'); axs[1].set_ylabel(r'$\dot{x}(t)$')
axs[1].set_ylim(-1.5, 1.5)
axs[1].set_xlim(-0.25, 0.99)
axs[1].grid(True)

plt.tight_layout()
plt.show()



