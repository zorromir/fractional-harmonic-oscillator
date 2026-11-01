# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def mittag_leffler_series(z, alpha, beta=1, terms=100):
    """Compute E_{alpha,beta}(z) via power series expansion."""
    result = np.zeros_like(z, dtype=np.float64)
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    return result

# Time values
t = np.linspace(20, 25, 500)

# Plot E_{alpha,1}(-t^alpha) for alpha in (1, 2)
plt.figure(figsize=(10, 6))

alphas = [0.5, 0.7, 0.9, 1.0]
for alpha in alphas:
    z = t**alpha
    y = mittag_leffler_series(z, alpha)
    plt.plot(t, y, label=fr"$\alpha={alpha}$")

plt.xlabel(r"$t$")
plt.ylabel(r"$E_{\alpha,1}(-t^\alpha)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()