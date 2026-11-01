import numpy as np
from numpy.testing import assert_allclose
import pytest
from fractional_oscillator import fractional_oscillator_exact


def test_initial_conditions():
    t=np.array([0.,1e-6])
    x=fractional_oscillator_exact(t,alpha=1.7,x0=2,v0=.5)
    assert x[0]==pytest.approx(2)
    assert (x[1]-x[0])/1e-6==pytest.approx(.5,abs=1e-4)


def test_classical_solution():
    t=np.linspace(0,10,50)
    expected=2*np.cos(1.4*t)+.3/1.4*np.sin(1.4*t)
    assert_allclose(fractional_oscillator_exact(t,alpha=2,omega=1.4,x0=2,v0=.3),expected)


def test_alpha_controls_periodic_vs_decay():
    # A fixed-time comparison, not a general monotonicity theorem.
    assert abs(fractional_oscillator_exact(10,alpha=1.5)) < abs(fractional_oscillator_exact(10,alpha=2))
