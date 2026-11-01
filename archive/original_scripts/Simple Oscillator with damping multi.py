# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:58:53 2025

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
m = 1.0  # Massa
k = 1.0  # Veerconstante

# Dempingwaarden en hun labels
damping_values = [0.05, 2, 5]  # Onder-, kritisch en overgedempt
labels = [
    r" (ondergedempt)", 
    r"(kritisch gedempt)", 
    r"(overgedempt)"
]
x0 = 1.0
v0 = 0.0
t_span = (0, 25)
t_eval = np.linspace(*t_span, 1000)

# Plot instellingen
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})

# Contrasterende kleuren
colors = ['black', 'darkred', 'darkblue']

# Oplossingen berekenen en tekenen
for gamma, label, color in zip(damping_values, labels, colors):
    def damped_oscillator(t, y):
        x, v = y
        dxdt = v
        dvdt = (-k * x - gamma * v) / m
        return [dxdt, dvdt]
    
    sol = solve_ivp(damped_oscillator, t_span, [x0, v0], t_eval=t_eval)
    plt.plot(sol.t, sol.y[0], label=label, linewidth=3, color=color)

# Plot opmaak
plt.xlabel("Tijd (t)")
plt.ylabel("Verplaatsing (x)")
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend()
plt.tight_layout()
plt.show()
