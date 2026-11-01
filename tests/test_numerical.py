import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gamma
from fractional_oscillator import caputo_l1,caputo_l2_quadratic,solve_oscillator,fractional_oscillator_exact


def test_l1_constant_and_linear():
    h=.01;t=np.linspace(0,1,101)
    assert_allclose(caputo_l1(np.ones_like(t),.6,h),0,atol=2e-14)
    exact=t[1:]**.4/gamma(1.4)
    assert_allclose(caputo_l1(t,.6,h)[1:],exact,atol=.005,rtol=.015)


def test_l1_quadratic_converges_under_refinement():
    alpha=.6;errors=[]
    for N in [50,100,200]:
        t=np.linspace(0,1,N+1)
        d=caputo_l1(t*t,alpha,1/N)
        errors.append(abs(d[-1]-2/gamma(3-alpha)))
    assert errors[2] < errors[1] < errors[0]


def test_quadratic_scheme_exact_for_quadratic_values():
    t=np.linspace(0,1,101)
    x=3+2*t+5*t*t
    d=caputo_l2_quadratic(x,1.6,.01,initial_velocity=2)
    truth=10*t[1:]**.4/gamma(1.4)
    assert_allclose(d[1:],truth,atol=1e-9,rtol=1e-9)


def test_quadratic_scheme_annihilates_constant_and_linear():
    t=np.linspace(0,1,101)
    for x,v in [(3+0*t,0),(2*t+3,2)]:
        d=caputo_l2_quadratic(x,1.5,.01,initial_velocity=v)
        assert_allclose(d,0,atol=1e-10)


def test_simple_oscillator_approximates_analytic_solution():
    t,x=solve_oscillator(alpha=1.8,beta=.6,damping=0,n_steps=500,t_end=5)
    exact=fractional_oscillator_exact(t,alpha=1.8)
    assert np.max(np.abs(x-exact)) < .045


def test_nonlinear_and_driven_are_finite():
    t,x=solve_oscillator(alpha=1.8,omega=1,restoring=np.sin,n_steps=150,t_end=4)
    assert len(t)==len(x)==151 and np.all(np.isfinite(x))
    t,y=solve_oscillator(alpha=1.8,beta=.6,damping=.4,n_steps=150,t_end=4,forcing=np.cos)
    assert np.all(np.isfinite(y))


def test_order_restrictions():
    with pytest.raises(ValueError):solve_oscillator(alpha=1.0)
    with pytest.raises(ValueError):caputo_l1([1,2],1.2,1)
