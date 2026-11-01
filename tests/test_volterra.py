import numpy as np
from fractional_oscillator import solve_one_term_volterra, fractional_oscillator_exact


def test_volterra_oscillator_accurate_over_fractional_orders():
    for alpha in [1.2,1.5,1.8,2.0]:
        t,x=solve_one_term_volterra(alpha=alpha,t_end=5,n_steps=250)
        error=np.max(np.abs(x-fractional_oscillator_exact(t,alpha=alpha)))
        assert error < .012, (alpha,error)


def test_volterra_convergence():
    errs=[]
    for n in (50,100,200):
        t,x=solve_one_term_volterra(alpha=1.4,t_end=5,n_steps=n)
        errs.append(np.max(np.abs(x-fractional_oscillator_exact(t,alpha=1.4))))
    assert errs[2]<errs[1]<errs[0]


def test_volterra_nonlinear_and_driven():
    t,x=solve_one_term_volterra(alpha=1.5,n_steps=200,t_end=5,restoring=np.sin,forcing=np.cos)
    assert np.all(np.isfinite(x)) and np.isfinite(t).all()
