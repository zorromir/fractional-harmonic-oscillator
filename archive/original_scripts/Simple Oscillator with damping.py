# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:52:11 2025

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
m = 1.0  # Mass
k = 1.0  # Spring constant
gamma = -0.1  # Damping coefficient


# Define the system of ODEs without the external driving force: dx/dt = v, dv/dt = (-kx - gamma*v)/m
def damped_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-k * x - gamma * v) / m  # Newton's second law without external force
    return [dxdt, dvdt]

# Initial conditions
x0 = 1.0  # Initial position
v0 = 0.0  # Initial velocity
t_span = (0, 50)  # Time span
t_eval = np.linspace(*t_span, 1000)  # Time points for solution

# Solve the system numerically using Runge-Kutta (RK45)
sol_damped = solve_ivp(damped_oscillator, t_span, [x0, v0], t_eval=t_eval)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sol_damped.t, sol_damped.y[0], label=r'$x(t)$ - Damped Oscillator', color='r')
plt.xlabel("Time (t)")
plt.ylabel("Displacement (x)")
plt.title("Damped Harmonic Oscillator (No External Force)")
plt.legend()
plt.grid()
plt.show()
