# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:15:32 2025

@author: kelvi
"""

# Interactive damped harmonic oscillator time-domain plot with sliders for ω and γ

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Time setup
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
Z0 = [1.0, 0.0]  # Initial conditions: x(0) = 1, v(0) = 0

# Create the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)

# Initial values
omega_init = 1.0
gamma_init = 0.2

# Solve ODE
def solve_damped_oscillator(omega, gamma):
    return solve_ivp(lambda t, z: [z[1], -2 * gamma * z[1] - omega**2 * z[0]],
                     t_span, Z0, t_eval=t_eval)

# Initial plot
sol = solve_damped_oscillator(omega_init, gamma_init)
[line] = ax.plot(sol.t, sol.y[0], 'b')
ax.set_title("Damped Harmonic Oscillator: Time Domain")
ax.set_xlabel("Time t")
ax.set_ylabel("Displacement x(t)")
ax.grid()

# Create sliders
ax_omega = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_gamma = plt.axes([0.25, 0.1, 0.65, 0.03])
s_omega = Slider(ax_omega, 'ω', 0.1, 3.0, valinit=omega_init, valstep=0.1)
s_gamma = Slider(ax_gamma, 'γ', 0.0, 2.0, valinit=gamma_init, valstep=0.05)

# Update function
def update(val):
    omega = s_omega.val
    gamma = s_gamma.val
    sol = solve_damped_oscillator(omega, gamma)
    line.set_data(sol.t, sol.y[0])
    ax.set_ylim(np.min(sol.y[0]) - 0.1, np.max(sol.y[0]) + 0.1)
    fig.canvas.draw_idle()

s_omega.on_changed(update)
s_gamma.on_changed(update)
plt.show()
