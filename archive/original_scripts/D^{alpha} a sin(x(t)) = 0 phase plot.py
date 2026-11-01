# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:41:19 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Parameters
T = 20
N = 250
dt = T / N
t = np.linspace(0, T, N + 1)
x0 = 1.0
x1 = 0.0

# L2 weights for Caputo derivative (1 < alpha < 2)
def l2_weights(alpha, n, dt):
    W = np.zeros(n + 2)
    for k in range(-1, n + 1):
        idx = k + 1
        if k == -1: W[idx] = 1
        elif k == 0: W[idx] = 2**(2 - alpha) - 3
        elif 1 <= k <= n - 2:
            W[idx] = (k + 2)**(2 - alpha) - 3*(k + 1)**(2 - alpha) + 3*k**(2 - alpha) - (k - 1)**(2 - alpha)
        elif k == n - 1:
            W[idx] = -2*n**(2 - alpha) + 3*(n - 1)**(2 - alpha) - (n - 2)**(2 - alpha)
        elif k == n:
            W[idx] = n**(2 - alpha) - (n - 1)**(2 - alpha)
    return W * dt**(-alpha) / gamma(3 - alpha)

# Solve the fractional nonlinear oscillator
def solve_fractional_nonlinear(alpha):
    x = np.zeros(N + 1)
    x[0] = x0
    x[1] = x0 + dt * x1

    for n in range(1, N):
        W = l2_weights(alpha, n, dt)
        sum_l2 = sum(W[k + 1] * x[n - k] for k in range(n + 1))
        nonlinear_term = np.sin(x[n])
        x[n + 1] = (-nonlinear_term - sum_l2) / W[0]
    return x

# Solve for two values of alpha
alphas = [1.2, 1.8]
results = [(alpha, solve_fractional_nonlinear(alpha)) for alpha in alphas]

# Plot x(t)
fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5))
for i, (alpha, x) in enumerate(results):
    axs1[i].plot(t, x)
    axs1[i].set_title(rf'Solution $x(t)$ for $\alpha = {alpha}$')
    axs1[i].set_xlabel('t')
    axs1[i].set_ylabel('x(t)')
    axs1[i].grid(True)
plt.tight_layout()
plt.show()

# Plot phase portraits with initial points skipped
skip = 50  # Number of initial points to skip

fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))
for i, (alpha, x) in enumerate(results):
    dx = np.gradient(x, dt)
    axs2[i].plot(x[skip:], dx[skip:])  # Skip the first `skip` points
    axs2[i].set_title(rf'Phase Portrait for $\alpha = {alpha}$')
    axs2[i].set_xlabel('x(t)')
    axs2[i].set_ylabel('dx/dt')
    axs2[i].grid(True)
plt.tight_layout()
plt.show()
