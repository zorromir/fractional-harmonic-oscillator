# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:05:05 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gamma

def mittag_leffler_series(z, alpha, beta=1, terms=300):
    """Compute E_{alpha,beta}(z) via power series expansion."""
    result = np.zeros_like(z, dtype=np.float64)
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    return result

# Grid for t and alpha
t_vals = np.linspace(0.1, 10, 200)
alpha_vals = np.linspace(0.35, 1.0, 100)
T, A = np.meshgrid(t_vals, alpha_vals)

# Compute z = -t^alpha and then E_{alpha,1}(z)
Z = -T**A
E = np.zeros_like(Z)

for i in range(len(alpha_vals)):
    E[i, :] = mittag_leffler_series(Z[i, :], alpha_vals[i])

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, A, E, cmap=cm.viridis, alpha=0.9)

# Highlight the curve for alpha = 1 in red
alpha_target = 1.0
idx_alpha_1 = np.argmin(np.abs(alpha_vals - alpha_target))
ax.plot(t_vals, alpha_vals[idx_alpha_1] * np.ones_like(t_vals), E[idx_alpha_1, :],
        color='red', linewidth=2.5)

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$x(t)$')
ax.legend()

# Set view direction
ax.view_init(elev=30, azim=155)
plt.tight_layout()
plt.show()
