# -*- coding: utf-8 -*-
"""
Created on Sun May 25 17:40:23 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Parameters
omega = 1.0
x0, x1 = 1.0, 1.0
T = 30
N = 1000
dt = T / N
t_vals = np.linspace(0, T, N+1)
alphas = [1.1, 1.3, 1.5, 1.9]

# Mittag-Leffler series expansion
def mittag_leffler(z, alpha, beta=1, terms=100):
    result = np.zeros_like(z)
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    return result

# Store results for plotting
results = []

for alpha in alphas:
    # Analytic solution with Mittag-Leffler functions
    z = -omega**2 * t_vals**alpha
    x = x0 * mittag_leffler(z, alpha, 1) + x1 * t_vals * mittag_leffler(z, alpha, 2)

    # Caputo-L1 approximation for D_t^alpha x(t)
    D_alpha_x = np.zeros_like(x)
    for n in range(2, len(t_vals)):  # start at n=2 to avoid zero to negative power
        acc = 0.0
        for k in range(n - 1):
            if n - k - 1 <= 0:
                continue
            coeff = (n - k)**(1 - alpha) - (n - k - 1)**(1 - alpha)
            acc += (x[k+1] - x[k]) * coeff
        D_alpha_x[n] = acc / (dt**alpha * gamma(2 - alpha))

    results.append((x, D_alpha_x, alpha))

# Plotting each (x(t), D_t^alpha x(t)) pair as a 2D subplot in a grid layout
num_plots = len(results)
cols = 2
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True, sharey=True)

for idx, (x, D_alpha_x, alpha) in enumerate(results):
    ax = axes.flat[idx]
    ax.plot(x, D_alpha_x, label=fr'$\alpha={alpha}$', color='navy')
    ax.set_title(fr'$\alpha={alpha}$')
    ax.set_xlabel(r'$x(t)$')
    ax.set_ylabel(r'$D_t^\alpha x(t)$')
    ax.grid(True)
    ax.legend()

# Hide any unused subplots
for j in range(idx + 1, rows * cols):
    fig.delaxes(axes.flat[j])

fig.suptitle('Fractional Phase Space Projections ($1 < \\alpha < 2$)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
