# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:27:51 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
m = 1.0  # mass (kg)
b = 0.1  # damping coefficient (Ns/m)
k = 20.0 # spring constant (N/m)
F0 = 1.0 # amplitude of driving force (N)
omega_d = 2.0 # driving frequency (rad/s)

# Differential equation for the driven harmonic oscillator
def oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (F0 * np.cos(omega_d * t) - b * v - k * x) / m
    return [dxdt, dvdt]

# Initial conditions
x0 = 0.0  # initial position
v0 = 0.0  # initial velocity
initial_conditions = [x0, v0]

# Time span for the simulation
t = np.linspace(0, 70, 2000)  # simulate from t=0 to t=50 seconds

# Solve the differential equation
solution = solve_ivp(oscillator, [t[0], t[-1]], initial_conditions, t_eval=t)

# Plot the solution (time response)
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='Position (x(t))')
plt.title('The Driven Harmonic Oscillator')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.grid(True)
plt.legend()

# Plot the phase plot (position vs. velocity)
plt.figure(figsize=(10, 5))
plt.plot(solution.y[0], solution.y[1], label='Phase Plot')
plt.title('Phase Plot of the Driven Harmonic Oscillator')
plt.xlabel('Position (x)')
plt.ylabel('Velocity (v)')
plt.grid(True)
plt.legend()
plt.show()
