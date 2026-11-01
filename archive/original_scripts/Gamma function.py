# -*- coding: utf-8 -*-
"""
Created on Fri May  2 08:43:53 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define x values (avoid x = 0 due to singularity)
x = np.linspace(0.02, 5, 500)
y = gamma(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$\Gamma(x)$", color="blue")
plt.xlabel("x")
plt.ylabel(r"$\Gamma(x)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
