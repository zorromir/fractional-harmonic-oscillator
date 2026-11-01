# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:11:08 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
k = 1  # Spring constant
m = 1  # Mass
omega = np.sqrt(k / m)  # Angular frequency

# Define the system of ODEs
def harmonic_oscillator(x, v, omega):
    dxdt = v
    dvdt = -omega**2 * x
    return dxdt, dvdt

# Generate a grid of initial conditions
x_vals = np.linspace(-2, 2, 20)
v_vals = np.linspace(-2, 2, 20)
X, V = np.meshgrid(x_vals, v_vals)

# Compute derivatives at each grid point
dX, dV = harmonic_oscillator(X, V, omega)

# Plot the phase portrait
plt.figure(figsize=(8, 6))
plt.quiver(X, V, dX, dV, color='blue', angles='xy', scale_units='xy', scale=10)

# Plot example trajectories
t = np.linspace(0, 10, 200)
for x0, v0 in [(-1, 1), (1, -1), (-2, 0), (0, 2)]:
    x_traj = x0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)
    v_traj = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    plt.plot(x_traj, v_traj, 'r')

# Labels and title
plt.xlabel('Position (x)')
plt.ylabel('Velocity (v)')
plt.title('Phase Portrait of the Simple Harmonic Oscillator')
plt.grid()
plt.show()
