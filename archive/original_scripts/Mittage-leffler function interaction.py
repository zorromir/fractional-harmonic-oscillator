# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:32:04 2025

@author: kelvi
"""

# Interactive plot of Mittag-Leffler E_{α,β}(z) with fixed axis limits

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import gamma

# Mittag-Leffler approximation
def mittag_leffler_ab(z, alpha, beta, terms=100):
    result = 0
    for k in range(terms):
        result += z**k / gamma(alpha * k + beta)
    return result

# z range and initial parameters
z_vals = np.linspace(0, 20, 800)
alpha_init = 1.0
beta_init = 1.0
y_vals = [mittag_leffler_ab(z, alpha_init, beta_init) for z in z_vals]

# Plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
[line] = ax.plot(z_vals, y_vals, label=r"$E_{\alpha,\beta}(z)$")

# Set fixed axis limits
ax.set_xlim(0, 20)
ax.set_ylim(0, 100)

ax.set_title("Generalized Mittag-Leffler Function $E_{\\alpha,\\beta}(z)$")
ax.set_xlabel("z")
ax.set_ylabel(r"$E_{\alpha,\beta}(z)$")
ax.grid()
ax.legend()

# Sliders
ax_alpha = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_beta = plt.axes([0.25, 0.1, 0.65, 0.03])
s_alpha = Slider(ax_alpha, r'$\alpha$', 0.1, 2.0, valinit=alpha_init, valstep=0.1)
s_beta = Slider(ax_beta, r'$\beta$', 0.1, 2.0, valinit=beta_init, valstep=0.1)

# Update function
def update(val):
    alpha = s_alpha.val
    beta = s_beta.val
    y_new = [mittag_leffler_ab(z, alpha, beta) for z in z_vals]
    line.set_ydata(y_new)
    fig.canvas.draw_idle()

s_alpha.on_changed(update)
s_beta.on_changed(update)
plt.show()
