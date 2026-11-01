"""Post-thesis Volterra product-integration solver for cross-checking."""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.special import gamma
from scipy.optimize import root_scalar


def solve_one_term_volterra(*, alpha: float, omega: float=1.0, x0: float=1.0,
                            v0: float=0.0, t_end: float=10.0, n_steps: int=400,
                            restoring: Callable[[float],float]|None=None,
                            forcing: Callable[[float],float]|None=None):
    """Solve a one-term Caputo equation by implicit product integration."""
    if not (1<alpha<=2) or not isinstance(n_steps,int) or n_steps<3:
        raise ValueError('Requires 1<alpha<=2 and an integer n_steps>=3.')
    if any(not np.isfinite(v) for v in (omega,x0,v0,t_end)) or omega<=0 or t_end<=0:
        raise ValueError('Requires finite omega>0, x0, v0, t_end>0.')
    g=(lambda y:y) if restoring is None else restoring
    f=(lambda t:0.0) if forcing is None else forcing
    t=np.linspace(0,t_end,n_steps+1);h=t_end/n_steps
    k=np.arange(1,n_steps+1,dtype=float)
    delta=(k**alpha-(k-1)**alpha)/alpha
    right=k*delta - (k**(alpha+1)-(k-1)**(alpha+1))/(alpha+1)
    left=delta-right
    c=h**alpha/gamma(alpha)
    left*=c;right*=c
    x=np.empty(n_steps+1);x[0]=x0
    vals=np.empty(n_steps+1);vals[0]=float(f(0))-omega**2*float(g(x0))
    for n in range(1,n_steps+1):
        # Every interval contributes its left-end forcing, previous intervals
        # additionally contribute the right-end forcing. The current right
        # endpoint is implicit and solved below.
        known=np.dot(left[:n],vals[n-1::-1])
        if n>1:known+=np.dot(right[1:n],vals[n-1:0:-1])
        base=x0+v0*t[n]+known+right[0]*float(f(float(t[n])))
        if restoring is None:
            x[n]=base/(1+right[0]*omega**2)
        else:
            fun=lambda y:y+right[0]*omega**2*float(g(y))-base
            try:
                sol=root_scalar(fun,x0=x[n-1],x1=x[n-1]+.01,method='secant',xtol=1e-12)
                if not sol.converged:raise RuntimeError('Nonconvergence')
                x[n]=sol.root
            except (RuntimeError,ValueError,ZeroDivisionError):
                M=max(5,2*abs(base)+5)
                while fun(-M)*fun(M)>0 and M<1e12:M*=2
                x[n]=root_scalar(fun,bracket=[-M,M],method='brentq').root
        vals[n]=float(f(float(t[n])))-omega**2*float(g(x[n]))
    return t,x
