# -*- coding: utf-8 -*-
"""
Created on Sat May 24 15:53:12 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Function to compute L1 Caputo derivative of t^p for given alpha
def caputo_l1_power(alpha, p, T, N):
    dt = T / N
    t = np.linspace(0, T, N+1)
    x = t**p
    D_alpha_x = np.zeros(N+1)

    # Precompute L1 weights
    b = np.array([((k + 1)**(1 - alpha) - k**(1 - alpha)) / gamma(2 - alpha) for k in range(N)])

    for n in range(1, N + 1):
        sum_term = 0
        for k in range(n):
            sum_term += b[n - k - 1] * (x[k + 1] - x[k])
        D_alpha_x[n] = sum_term * dt**(-alpha)

    return t, D_alpha_x

# Parameters
T = 6
N = 300
p = 2  # exponent for t^p
alphas = np.linspace(0.1, 1.0, 50)
t_grid = np.linspace(0, T, N+1)
alpha_grid, time_grid = np.meshgrid(alphas, t_grid)
result_grid = np.zeros_like(alpha_grid)

# Compute Caputo derivative for each alpha
for i, alpha in enumerate(alphas):
    _, D_alpha_power = caputo_l1_power(alpha, p, T, N)
    result_grid[:, i] = D_alpha_power

# Compute f(t) = 2t to overlay
line_t = t_grid
line_f = 2 * t_grid
line_alpha = 0 * t_grid + 1  # fixed alpha slice

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface for Caputo derivatives
ax.plot_surface(time_grid, alpha_grid, result_grid, cmap='viridis', alpha=0.9)

# Overlay line f(t) = 2t in red at alpha = 0
ax.plot(line_t, line_alpha, line_f, color='red', linewidth=2, label=r'$f(t) = 2t$')

# Labels and formatting
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(rf'$D^\alpha t^{p}$')
ax.set_title(rf'Caputo Derivative of $t^{p}$ with $f(t) = 2t$ Overlay at $\alpha=0$')
ax.legend()
plt.tight_layout()
plt.show()
