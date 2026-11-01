# -*- coding: utf-8 -*-
"""
Created on Sun May 25 12:15:09 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gamma

def mittag_leffler_series(z, alpha, beta=1, terms=100):
    """Compute E_{alpha,beta}(z) via power series expansion."""
    result = np.zeros_like(z, dtype=np.float64)
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    return result

# Parameters
x_0 = 1
x_1 = 1

# Grid for t and alpha
t_vals = np.linspace(0.0, 25, 1000)
alpha_vals = np.linspace(1.05, 1.95, 1000)
T, A = np.meshgrid(t_vals, alpha_vals)

# Compute z = -t^alpha
Z = -1*T**A

# Compute both Mittag-Leffler terms
E_alpha_1 = np.zeros_like(Z)
E_alpha_2 = np.zeros_like(Z)

for i in range(len(alpha_vals)):
    E_alpha_1[i, :] = mittag_leffler_series(Z[i, :], alpha_vals[i], beta=1)
    E_alpha_2[i, :] = mittag_leffler_series(Z[i, :], alpha_vals[i], beta=2)

# Final solution: x(t) = x_0 E_{alpha,1}(-t^alpha) + x_1 * t * E_{alpha,2}(-t^alpha)
X = x_0 * E_alpha_1 + x_1 * T * E_alpha_2

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, A, X, cmap=cm.viridis, alpha=0.9)

# Highlight the curve for alpha = 1 in red
alpha_target = 2.0
idx_alpha_1 = np.argmin(np.abs(alpha_vals - alpha_target))
ax.plot(t_vals, alpha_vals[idx_alpha_1] * np.ones_like(t_vals), X[idx_alpha_1, :],
        color='red', linewidth=2.5)

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$x(t)$')
ax.legend()

# Set view direction
ax.view_init(elev=30, azim=155)
plt.tight_layout()
plt.show()

