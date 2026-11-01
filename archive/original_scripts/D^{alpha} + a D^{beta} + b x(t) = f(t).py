import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Parameters
T = 20
N = 1000
dt = T / N
t = np.linspace(0, T, N + 1)
omega = 1.0
x0 = 1.0
x1 = 0.0

# L2 weights function
def l2_weights(alpha, n, dt):
    W = np.zeros(n + 2)
    for k in range(-1, n + 1):
        idx = k + 1
        if k == -1:
            W[idx] = 1
        elif k == 0:
            W[idx] = 2**(2 - alpha) - 3
        elif 1 <= k <= n - 2:
            W[idx] = (k + 2)**(2 - alpha) - 3 * (k + 1)**(2 - alpha) + 3 * k**(2 - alpha) - (k - 1)**(2 - alpha)
        elif k == n - 1:
            W[idx] = -2 * n**(2 - alpha) + 3 * (n - 1)**(2 - alpha) - (n - 2)**(2 - alpha)
        elif k == n:
            W[idx] = n**(2 - alpha) - (n - 1)**(2 - alpha)
    return W * dt**(-alpha) / gamma(3 - alpha)

# L2 solver
def solve_fde(alpha):
    x = np.zeros(N + 1)
    x[0] = x0
    x[1] = x0 + dt * x1
    for n in range(1, N):
        W = l2_weights(alpha, n, dt)
        sum_l2 = sum(W[k + 1] * x[n - k] for k in range(n + 1))
        x[n + 1] = (-omega**2 * x[n] - sum_l2) / W[0]
    return x

# Numerical derivative for phase portrait
def compute_derivative(x, dt):
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

# Run for different alpha values
alphas = [1.4, 1.8]
results = [(alpha, solve_fde(alpha)) for alpha in alphas]
x_classic = np.cos(omega * t)

# Plot separate time-domain figures
plt.rcParams.update({'font.size': 14})
for alpha, x in results:
    plt.figure(figsize=(8, 5))
    plt.plot(t, x, label=rf"Fractionele oscillator ($\alpha = {alpha}$)", linewidth=3, color='black')
    plt.plot(t, x_classic, label="Klassieke oscillator", linestyle='--', color='gray', linewidth=2)
    plt.title(rf"Oplossing $x(t)$ voor $\alpha = {alpha}$")
    plt.xlabel("Tijd $t$")
    plt.ylabel("Verplaatsing $x(t)$")
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot phase portraits side by side
trim = 5
fig, axs = plt.subplots(1, len(alphas), figsize=(14, 5))

for i, (alpha, x) in enumerate(results):
    dx = compute_derivative(x, dt)
    x_trimmed = x[trim:-trim]
    dx_trimmed = dx[trim:-trim]
    axs[i].plot(x_trimmed, dx_trimmed, color='blue', linewidth=2, label=rf"$\alpha = {alpha}$")
    axs[i].set_title(rf"phase portret for $\alpha = {alpha}$")
    axs[i].set_xlabel(r"$x(t)$")
    axs[i].set_ylabel(r"$dx/dt$")
    axs[i].grid(True, linestyle='--', linewidth=0.7)
    axs[i].legend()

plt.tight_layout()
plt.show()
