# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:06:36 2025

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Time settings
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
Z0 = [1.0, 0.0]  # Initial condition: x(0) = 1, v(0) = 0

# Figure setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

# Grid for phase portrait
x_vals = np.linspace(-2, 2, 20)
v_vals = np.linspace(-3, 3, 20)
X, V = np.meshgrid(x_vals, v_vals)

# Initial parameters
omega_init = 1.0
gamma_init = 0.2

# System solver
def solve_damped(omega, gamma):
    return solve_ivp(lambda t, z: [z[1], -2 * gamma * z[1] - omega**2 * z[0]], t_span, Z0, t_eval=t_eval)

# Initial vector field and trajectory
U = V
W = - gamma_init * V - omega_init**2 * X
quiv = ax.quiver(X, V, U, W, color='purple', alpha=0.3)

sol = solve_damped(omega_init, gamma_init)
traj, = ax.plot(sol.y[0], sol.y[1], 'm')

# Labels
ax.set_title("Damped Harmonic Oscillator Phase Portrait")
ax.set_xlabel("x")
ax.set_ylabel("dx/dt")
ax.grid()

# Sliders
ax_omega = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_gamma = plt.axes([0.25, 0.10, 0.65, 0.03])
s_omega = Slider(ax_omega, 'ω', 0.1, 3.0, valinit=omega_init, valstep=0.1)
s_gamma = Slider(ax_gamma, 'γ', 0.0, 1.5, valinit=gamma_init, valstep=0.05)

# Update function
def update(val):
    omega = s_omega.val
    gamma = s_gamma.val
    U = V
    W = -2 * gamma * V - omega**2 * X
    quiv.set_UVC(U, W)
    sol = solve_damped(omega, gamma)
    traj.set_data(sol.y[0], sol.y[1])
    fig.canvas.draw_idle()

s_omega.on_changed(update)
s_gamma.on_changed(update)
plt.show()
