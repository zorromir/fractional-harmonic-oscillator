# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:43:54 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# L2 method for Caputo derivative of sin(t) using definition from thesis

def caputo_l2_sin(alpha, T, N):
    dt = T / N
    t = np.linspace(0, T, N+1)
    x = np.sin(t)
    D_alpha_x = np.zeros(N+1)

    factor = dt**(-alpha) / gamma(3 - alpha)

    for n in range(2, N+1):
        sum_term = 0.0
        for k in range(-1, n+1):
            if k == -1:
                W = 1
            elif k == 0:
                W = 2**(2 - alpha) - 3
            elif 1 <= k <= n - 2:
                W = (k + 2)**(2 - alpha) - 3 * (k + 1)**(2 - alpha) + \
                    3 * k**(2 - alpha) - (k - 1)**(2 - alpha)
            elif k == n - 1:
                W = -2 * n**(2 - alpha) + 3 * (n - 1)**(2 - alpha) - (n - 2)**(2 - alpha)
            elif k == n:
                W = n**(2 - alpha) - (n - 1)**(2 - alpha)
            else:
                continue

            if 0 <= n - k < len(x):
                sum_term += W * x[n - k]

        D_alpha_x[n] = factor * sum_term

    return t, D_alpha_x

# Parameters
T = 12
N = 300
alphas = np.linspace(1.05, 1.95, 30)
cut_start = 10   # truncate initial points
cut_end = 280    # truncate last points for stability in plot
t_grid = np.linspace(0, T, N+1)[cut_start:cut_end]
alpha_grid, time_grid = np.meshgrid(alphas, t_grid)
result_grid = np.zeros_like(alpha_grid)

# Compute Caputo derivative for each alpha using L2 method
for i, alpha in enumerate(alphas):
    _, D_alpha_sin = caputo_l2_sin(alpha, T, N)
    result_grid[:, i] = D_alpha_sin[cut_start:cut_end]

# 3D Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(time_grid, alpha_grid, result_grid, cmap='viridis')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$D^\alpha \sin(t)$')
ax.set_title(r'Caputo Derivative of $\sin(t)$ using L2 Method for $1 < \alpha < 2$')
plt.tight_layout()
plt.show()
