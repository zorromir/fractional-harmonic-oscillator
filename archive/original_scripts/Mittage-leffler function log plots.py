# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:30:05 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Mittag-Leffler approximation
def mittag_leffler_ab(z, alpha, beta, terms=100):
    result = 0
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    return result

# z range
z_vals = np.linspace(0.1, 20, 800)

# Parameter combinations
alpha_beta_pairs = [(0.3, 0.5), (0.5, 1.0), (1.0, 1.0), (1.5, 0.5), (2.0, 1.5)]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for alpha, beta in alpha_beta_pairs:
    y_vals = [mittag_leffler_ab(z, alpha, beta) for z in z_vals]
    ax.plot(z_vals, y_vals, label=fr"$E_{{{alpha},{beta}}}(z)$")

ax.set_title("Mittag-Leffler Function $E_{\\alpha,\\beta}(z)$ (log scale)")
ax.set_xlabel("z")
ax.set_ylabel(r"$E_{\alpha,\beta}(z)$")
ax.set_yscale("log")  # Logarithmic y-axis
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.show()
