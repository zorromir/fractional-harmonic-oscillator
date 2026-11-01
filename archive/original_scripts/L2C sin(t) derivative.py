# -*- coding: utf-8 -*-
"""
Created on Sat May 24 15:58:42 2025

@author: kelvi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# L2C method for Caputo derivative of sin(t), 1 < alpha < 2
def caputo_l2c_sin(alpha, T, N):
    dt = T / N
    t = np.linspace(0, T, N+1)
    x = np.cos(t)
    D_alpha_x = np.zeros(N+1)

    # Compute omega weights for L2C method from k = -1 to k = n+1
    omega = np.zeros(N+3)  # shifted index: omega[k+1] corresponds to k
    for k in range(-1, N+2):
        idx = k + 1
        if k == -1:
            omega[idx] = 1
        elif k == 0:
            omega[idx] = 2**(2 - alpha) - 2
        elif k == 1:
            omega[idx] = 3**(2 - alpha) - 2 * 2**(2 - alpha)
        elif 2 <= k <= N - 1:
            omega[idx] = (k + 2)**(2 - alpha) - 2 * (k + 1)**(2 - alpha) + \
                           2 * (k - 1)**(2 - alpha) - (k - 2)**(2 - alpha)
        elif k == N:
            omega[idx] = - N**(2 - alpha) - (N - 3)**(2 - alpha) + 2 * (N - 2)**(2 - alpha)
        elif k == N + 1:
            omega[idx] = - N**(2 - alpha) + 2 * (N - 1)**(2 - alpha) - (N - 2)**(2 - alpha)

    factor = dt**(-alpha) / (2 * gamma(3 - alpha))
    for n in range(2, N+1):
        sum_term = 0
        for k in range(-1, n+2):
            idx = k + 1
            xi = n - k
            if 0 <= xi < len(x):
                sum_term += omega[idx] * x[xi]
        D_alpha_x[n] = factor * sum_term

    return t, D_alpha_x

# Parameters
T = 6
N = 500
alphas = np.linspace(1.1, 1.9, 1000)
cut_start = 20   # truncate initial points
cut_end = 490    # truncate last points for stability in plot
t_grid = np.linspace(0, T, N+1)[cut_start:cut_end]
alpha_grid, time_grid = np.meshgrid(alphas, t_grid)
result_grid = np.zeros_like(alpha_grid)

# Compute Caputo derivative for each alpha using L2C method
for i, alpha in enumerate(alphas):
    _, D_alpha_sin = caputo_l2c_sin(alpha, T, N)
    result_grid[:, i] = D_alpha_sin[cut_start:cut_end]

# 3D Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(time_grid, alpha_grid, result_grid, cmap='plasma')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$D^\alpha \sin(t)$')
ax.set_title(r'Caputo Derivative of $\sin(t)$ using L2C Method for $1 < \alpha < 2$')
plt.tight_layout()
plt.show()
