# The Fractional Harmonic Oscillator: Analysis and Computations

**Kelvin Schouten · Utrecht University · BSc Mathematics · 2025**  
**Supervisor:** Paul Zegeling

This is a short reading guide to my bachelor's thesis, **not the complete PDF manuscript**. The original manuscript has not yet been added to the public repository.

## Research question

How can fractional differential equations describe the behaviour of a fractional harmonic oscillator, and what are the consequences for numerical accuracy, stability and physical interpretation?

The thesis starts with ordinary oscillators, then replaces integer-order derivatives by fractional derivatives. This introduces memory and nonlocal effects: the dynamics can depend on earlier parts of the trajectory.

## Mathematical model

A central example is the Caputo fractional oscillator:

$$
{}^C D_t^\alpha x(t)+\omega^2x(t)=0,\qquad 1<\alpha\le2.
$$

For initial conditions $x(0)=x_0$ and $x'(0)=v_0$, its linear solution is expressed with Mittag–Leffler functions:

$$
x(t)=x_0E_{\alpha,1}(-\omega^2t^\alpha)
+v_0tE_{\alpha,2}(-\omega^2t^\alpha).
$$

When $\alpha=2$, this reduces to the classical harmonic oscillator. Fractional orders change the oscillation and decay behaviour.

## Thesis chapters

- **Classical harmonic oscillators:** simple, nonlinear, damped and driven models and their phase portraits.
- **Fractional calculus:** Gamma and Mittag–Leffler functions; Grünwald–Letnikov, Riemann–Liouville and Caputo derivatives.
- **Analytical solution methods:** Laplace transforms, fractional power series and Mittag–Leffler solutions.
- **Numerical methods:** L1 and L2 discretisations for Caputo derivatives and numerical tests.
- **Fractional oscillator models:** simple, nonlinear, damped and driven cases, with trajectory comparisons.
- **Further directions:** exploratory extensions to Van der Pol and Lotka–Volterra systems.

## Code and reproducibility

The repository keeps the [original thesis scripts](archive/original_scripts/) for reference, alongside [restructured functions](src/fractional_oscillator/) and [figure generation](examples/generate_figures.py). This newer codebase is a reconstruction for readability and demonstration, not an exact copy of the original 2025 working environment. The plots are numerical illustrations, not measured data.

[Research page](https://zorromir.github.io/#research) · [Numerical code](src/fractional_oscillator/)
