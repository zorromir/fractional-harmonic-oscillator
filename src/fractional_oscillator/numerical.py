"""Caputo difference schemes and oscillator simulations (thesis Chapters 5–6)."""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.special import gamma
from scipy.optimize import root_scalar


def _grid(values, step):
    arr=np.asarray(values,dtype=float)
    if arr.ndim!=1 or arr.size<2 or not np.all(np.isfinite(arr)) or not np.isfinite(step) or step<=0:
        raise ValueError('Expected finite 1D values of length >=2 and step >0.')
    return arr


def caputo_l1(values, alpha: float, step: float) -> np.ndarray:
    """Approximate a Caputo derivative of order 0 < alpha < 1 using L1 weights."""
    x=_grid(values,step)
    if not 0<alpha<1:raise ValueError('L1 requires 0 < alpha < 1.')
    n=x.size-1;k=np.arange(n,dtype=float)
    weights=(k+1)**(1-alpha)-k**(1-alpha)
    differences=np.diff(x)
    result=np.zeros(n+1,dtype=float)
    c=step**(-alpha)/gamma(2-alpha)
    for i in range(1,n+1):
        result[i]=c*np.dot(weights[:i],differences[i-1::-1])
    return result


def caputo_l2_quadratic(values, alpha: float, step: float, *, initial_velocity: float) -> np.ndarray:
    """Approximate a Caputo derivative using piecewise-quadratic interpolation."""
    x=_grid(values,step)
    if not 1<alpha<2:raise ValueError('Quadratic Caputo scheme requires 1 < alpha < 2.')
    if not np.isfinite(initial_velocity):raise ValueError('initial_velocity must be finite.')
    n=x.size-1;d2=np.empty(n)
    d2[0]=2*(x[1]-x[0]-step*initial_velocity)/(step*step)
    if n>1:d2[1:]=(x[2:]-2*x[1:-1]+x[:-2])/(step*step)
    k=np.arange(n,dtype=float)
    weights=(k+1)**(2-alpha)-k**(2-alpha)
    c=step**(2-alpha)/gamma(3-alpha)
    result=np.zeros(n+1,dtype=float)
    for i in range(1,n+1):
        result[i]=c*np.dot(weights[:i],d2[i-1::-1])
    return result


def solve_oscillator(*, alpha: float=1.8,beta:float=0.6,damping:float=0.0,
                     omega:float=1.0,x0:float=1.0,v0:float=0.0,
                     t_end:float=10.0,n_steps:int=400,
                     forcing:Callable[[float],float]|None=None,
                     restoring:Callable[[float],float]|None=None):
    """Solve a fractional oscillator with optional damping, forcing and nonlinearity."""
    if not 1<alpha<=2 or not 0<beta<=1 or not isinstance(n_steps,int) or n_steps<3:
        raise ValueError('Requires 1 < alpha <= 2, 0 < beta <= 1, and n_steps >= 3.')
    if any(not np.isfinite(v) for v in [damping,omega,x0,v0,t_end]) or damping<0 or omega<=0 or t_end<=0:
        raise ValueError('Expected finite coefficients, damping>=0, omega>0, t_end>0.')
    f=(lambda t: 0.0) if forcing is None else forcing
    g=(lambda x: x) if restoring is None else restoring
    h=t_end/n_steps
    t=np.linspace(0,t_end,n_steps+1)
    x=np.zeros(n_steps+1);x[0]=x0
    ka=np.arange(n_steps,dtype=float)
    wa=(ka+1)**(2-alpha)-ka**(2-alpha) if alpha<2 else np.r_[1.0,np.zeros(n_steps-1)]
    ca=h**(-alpha)/gamma(3-alpha)
    if beta<1:
        kb=np.arange(n_steps,dtype=float)
        wb=(kb+1)**(1-beta)-kb**(1-beta)
        cb=h**(-beta)/gamma(2-beta)
    else:
        wb=np.r_[1.0,np.zeros(n_steps-1)];cb=1/h
    # One time step's solution appears as the last unknown in each convolution.
    for n in range(1,n_steps+1):
        if n==1:
            acoef=2*ca
            aknown=-2*ca*(x0+h*v0)
        else:
            acoef=ca
            # first interval has an initial-velocity start-up correction
            aknown=2*ca*wa[n-1]*(x[1]-x0-h*v0)
            if n>2:
                second=x[2:n]-2*x[1:n-1]+x[:n-2]
                aknown+=ca*np.dot(wa[1:n-1],second[::-1])
            aknown+=ca*(-2*x[n-1]+x[n-2])
        if damping:
            bcoef=cb
            bknown=-cb*x[n-1]
            if n>1:
                bknown+=cb*np.dot(wb[1:n],np.diff(x[:n])[::-1])
        else:bcoef=bknown=0.0
        coefficient=acoef+damping*bcoef
        rhs=float(f(float(t[n])))-aknown-damping*bknown
        # monotone linear case can be solved without a nonlinear root-finder
        if restoring is None:
            x[n]=rhs/(coefficient+omega**2)
        else:
            fun=lambda y:coefficient*y+omega**2*float(g(y))-rhs
            # Derivative-free secant iteration, bracketing fallback.
            try:
                sol=root_scalar(fun,x0=x[n-1],x1=x[n-1]+0.01,method='secant',xtol=1e-11,maxiter=100)
                if not sol.converged or not np.isfinite(sol.root):raise ArithmeticError('Secant did not converge')
                x[n]=sol.root
            except (ArithmeticError,ValueError,RuntimeError,ZeroDivisionError):
                M=max(5.0,abs(rhs/coefficient)+5.0)
                for _ in range(20):
                    if fun(-M)*fun(M)<=0:break
                    M*=2
                else:raise ArithmeticError('Failed to bracket nonlinear root.')
                x[n]=root_scalar(fun,bracket=[-M,M],method='brentq').root
    return t,x
