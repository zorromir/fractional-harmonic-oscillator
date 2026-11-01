# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:20:04 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

# Define a function to compute the harmonic oscillator solution
def harmonic_oscillator(k, m, x0, x1, t):
    omega = np.sqrt(k / m)  # Natural frequency
    return x0 * np.cos(omega * t) + (x1 / omega) * np.sin(omega * t)

# Time array
t = np.linspace(0, 25, 1000)

# Different sets of (k, m, x0, x1) values
params = [
    {"k": 1, "m": 1, "x0": 1, "x1": 0},
]

# Plot different solutions
plt.figure(figsize=(10, 6))

for param in params:
    x_t = harmonic_oscillator(param["k"], param["m"], param["x0"], param["x1"], t)
    label = f"k={param['k']}, m={param['m']}, x0={param['x0']}, x1={param['x1']}"
    plt.plot(t, x_t, label=label, color='black', linewidth=3)  # expliciet zwart

# Formatting the plot
plt.axhline(1, color='black', linewidth=0.5)
plt.axvline(1, color='black', linewidth=0.5)
plt.xlabel("Tijd (t)")
plt.ylabel("verplaatsing (x)")
plt.title("Oplossing met $ \omega$ = 1 , $x_0$ = 1, $ x_1$ = 0 ")
plt.grid()
plt.show()