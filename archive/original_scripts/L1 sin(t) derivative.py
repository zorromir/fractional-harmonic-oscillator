# -*- coding: utf-8 -*-
"""
Created on Sat May 24 15:45:03 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Function to compute L1 Caputo derivative of sin(t) for given alpha
def caputo_l1_sin(alpha, T, N):
    dt = T / N
    t = np.linspace(0, T, N+1)
    x = np.sin(t)
    D_alpha_x = np.zeros(N+1)

    # Precompute L1 weights
    b = np.array([( (k+1)**(1 - alpha) - k**(1 - alpha) ) / gamma(2 - alpha) for k in range(N)])

    for n in range(1, N+1):
        sum_term = 0
        for k in range(n):
            sum_term += b[n - k - 1] * (x[k + 1] - x[k])
        D_alpha_x[n] = sum_term * dt**(-alpha)

    return t, D_alpha_x

# Parameters
T = 6
N = 400
alphas = np.linspace(0.1, 0.9, 80)
t_grid = np.linspace(0, T, N+1)
alpha_grid, time_grid = np.meshgrid(alphas, t_grid)
result_grid = np.zeros_like(alpha_grid)

# Compute Caputo derivative for each alpha
for i, alpha in enumerate(alphas):
    _, D_alpha_sin = caputo_l1_sin(alpha, T, N)
    result_grid[:, i] = D_alpha_sin

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(time_grid, alpha_grid, result_grid, cmap='viridis')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$D^\alpha \sin(t)$')
ax.set_title('Caputo Derivative of $\sin(t)$ for $0.2 \leq \\alpha \leq 1.0$ using L1 Method')
plt.tight_layout()
plt.show()
