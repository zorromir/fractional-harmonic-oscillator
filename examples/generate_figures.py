"""Generate the five thesis-inspired figures in figures/."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# The editable install is recommended; the fallback supports direct checkout.
try:
    from fractional_oscillator import fractional_oscillator_exact, solve_oscillator, solve_one_term_volterra
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from fractional_oscillator import fractional_oscillator_exact, solve_oscillator, solve_one_term_volterra

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

INK='#20343a';TEAL='#24766e';COPPER='#b98666';MUTED='#627f79';BG='#f8f9f5'

def setup():
    plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,
        'axes.edgecolor':'#a6b8af','axes.labelcolor':INK,'text.color':INK,
        'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#dbe4e0',
        'savefig.facecolor':BG})

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def plot_time(out):
    t=np.linspace(0,16,321)
    fig,ax=plt.subplots(figsize=(9.5,4.4))
    for a,c in [(1.2,MUTED),(1.5,COPPER),(1.8,TEAL),(2.0,INK)]:
        ax.plot(t,fractional_oscillator_exact(t,alpha=a),color=c,lw=2,label=fr'$\alpha={a:g}$')
    ax.set(xlabel='Time t',ylabel='Displacement x(t)',title='Simple fractional harmonic oscillator')
    ax.grid(alpha=.7);ax.legend(ncol=4);save(fig,out,'01_oscillator_time')

def plot_phase(out):
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.8))
    t=np.linspace(0,15,451)
    for ax,(a,c) in zip(axs,[(1.2,MUTED),(1.8,TEAL),(2.0,INK)]):
        x=fractional_oscillator_exact(t,alpha=a);v=np.gradient(x,t)
        ax.plot(x,v,lw=2,color=c);ax.set(title=fr'$\alpha={a:g}$',xlabel='Position x(t)',ylabel='dx/dt')
        ax.grid(alpha=.65);ax.axis('equal')
    fig.suptitle('Position–velocity projections (not complete fractional state spaces)')
    save(fig,out,'02_phase_projections')

def plot_memory(out):
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    lag=np.linspace(.1,8,250)
    for a,c in [(1.2,MUTED),(1.5,COPPER),(1.8,TEAL)]:
        ax.plot(lag,lag**(1-a)/gamma(2-a),color=c,lw=2,label=fr'$\alpha={a:g}$')
    ax.set(title='Caputo memory kernel (1 < α < 2)',xlabel='Elapsed time / lag u',ylabel=r'$u^{1-\alpha}/\Gamma(2-\alpha)$')
    ax.grid(alpha=.7);ax.legend();save(fig,out,'03_memory_kernel')

def plot_nonlinear(out):
    t,x=solve_one_term_volterra(alpha=1.8,omega=1,x0=1,v0=0,n_steps=650,t_end=15,restoring=np.sin)
    fig,ax=plt.subplots(figsize=(9.5,4.5));ax.plot(t,x,color=TEAL,lw=2)
    ax.set(title='Nonlinear fractional oscillator (Volterra product integration)',xlabel='Time t',ylabel='Displacement x(t)');ax.grid(alpha=.7)
    save(fig,out,'04_nonlinear_oscillator')

def plot_driven(out):
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for alpha,color in [(1.4,MUTED),(1.8,TEAL)]:
        t,x=solve_oscillator(alpha=alpha,beta=.6,damping=.6,omega=1,x0=1,v0=0,
                        t_end=20,n_steps=800,forcing=lambda t: np.cos(t))
        ax.plot(t,x,color=color,lw=2,label=fr'$\alpha={alpha:g}$')
    ax.set(title='Driven, damped fractional oscillator (revised solver)',xlabel='Time t',ylabel='Displacement x(t)')
    ax.grid(alpha=.7);ax.legend();save(fig,out,'05_driven_oscillator')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='figures',help='Output directory for .png and .svg images')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    setup()
    for function in (plot_time,plot_phase,plot_memory,plot_nonlinear,plot_driven):
        function(out)
    print(f'Generated five PNG/SVG pairs in {out.resolve()}')

if __name__=='__main__':main()
