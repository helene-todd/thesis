import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
import math as math
import cmath as cmath
import numpy as np
import os, csv
from pathlib import Path
import matplotlib

# ffmpeg -framerate 25 -i %01d.png output.avi

plt.rcParams['axes.xmargin'] = 0

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=16)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = plt.cm.plasma(np.linspace(0,0.75,3))

# General parameters
T = 1000
STEP = 1e-3
tau_m = 1
tau_d = 1
eta = 1
delta = 0.3
Jc = -8

def euler(r0, v0, s0, g, J) :
    v, r, s = [[np.mean(np.random.normal(v0[0], 1, 1000))], [np.mean(np.random.normal(v0[1], 1, 1000))]], [[r0[0]], [r0[1]]], [[s0[0]], [s0[1]]]
    for i in range(1, int(T/STEP)):
        r[0].append(r[0][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[0][i-1]*v[0][i-1] - g[0]*r[0][i-1])/tau_m)
        r[1].append(r[1][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[1][i-1]*v[1][i-1] - g[1]*r[1][i-1])/tau_m)
        v[0].append(v[0][i-1] + STEP*(v[0][i-1]**2 + eta + J[0]*tau_m*s[0][i-1] + Jc*tau_m*s[1][i-1] - (math.pi**2)*(tau_m*r[0][i-1])**2)/tau_m)
        v[1].append(v[1][i-1] + STEP*(v[1][i-1]**2 + eta + J[1]*tau_m*s[1][i-1] + Jc*tau_m*s[0][i-1] - (math.pi**2)*(tau_m*r[1][i-1])**2)/tau_m)
        s[0].append(s[0][i-1] + STEP*(-s[0][i-1] + r[0][i-1])/tau_d)
        s[1].append(s[1][i-1] + STEP*(-s[1][i-1] + r[1][i-1])/tau_d)
    return np.array(r[0][int(2*T/(3*STEP)):]), np.array(v[0][int(2*T/(3*STEP)):]), np.array(s[0][int(2*T/(3*STEP)):])

def get_equilibria(g, J) :
    # Computing equilibria
    def f(r):
        eq1 = delta**2/(2*tau_m*math.pi)**2 + (g[0]*r[0])**2/4 - delta*g[0]*r[0]/(2*tau_m*math.pi) + eta*r[0]**2 + J[0]*tau_d*r[0]**3 + Jc*tau_d*r[1]*r[0]**2 - (math.pi*tau_m*r[0]**2)**2
        eq2 = delta**2/(2*tau_m*math.pi)**2 + (g[1]*r[1])**2/4 - delta*g[1]*r[1]/(2*tau_m*math.pi) + eta*r[1]**2 + J[1]*tau_d*r[1]**3 + Jc*tau_d*r[0]*r[1]**2 - (math.pi*tau_m*r[1]**2)**2
        return np.array([eq1, eq2])

    r2 = np.linspace(0.01, 1, 1000)
    r1 = -delta**2/(Jc*tau_d*r2**2*4*tau_m**2*math.pi**2) - g[1]**2/(4*Jc*tau_d) + delta*g[1]/(Jc*tau_d*r2*2*tau_m*math.pi) - eta/(Jc*tau_d) - J[1]*tau_d*r2/(Jc*tau_d) + math.pi**2*tau_m*r2**2/(Jc*tau_d)
    y = delta**2/(2*tau_m*math.pi)**2 + (g[0]*r1)**2/4 - delta*g[0]*r1/(2*tau_m*math.pi) + eta*r1**2 + J[0]*tau_d*r1**3 + Jc*tau_d*r2*r1**2 - (math.pi*tau_m*r1**2)**2
    idx = np.argwhere(np.diff(np.sign(y))).flatten()

    r_e, v_e, s_e = [], [], []

    for el in idx :
        if r1[el] > 0 :
            r_e.append(fsolve(f, x0=[r1[el], r2[el]]))
            v_e.append([-delta/(2*r_e[-1][0]*tau_m*math.pi) + g[0]/2, -delta/(2*r_e[-1][1]*tau_m*math.pi) + g[1]/2])
            s_e.append([r_e[-1][0], r_e[-1][1]])

    return np.array(r_e), np.array(v_e), np.array(s_e)


def get_eigenvalues(g, J, r_e, v_e, s_e) :
    # Computing eigenvalues
    a = np.array([ [2*v_e[0]-g[0], 0, 2*r_e[0], 0, 0, 0 ],
                    [0, 2*v_e[1]-g[1], 0, 2*r_e[1], 0, 0 ],
                    [-2*math.pi**2*tau_m**2*r_e[0], 0, 2*v_e[0], 0, J[0]*tau_d, Jc*tau_d ],
                    [0, -2*math.pi**2*tau_m**2*r_e[1], 0, 2*v_e[1], Jc*tau_d, J[1]*tau_d ],
                    [1, 0, 0, 0, -1, 0 ],
                    [0, 1, 0, 0, 0, -1 ] ])
    w,v=eig(a)
    return w
        


J2 = np.linspace(-2, -6, 200)
g = [0.4, 2]
i = 0

for Js2 in J2:
    print(i)
    fig, ax = plt.subplots(3, 1, figsize=(4,8), sharex=True, sharey=True)
    fig.suptitle(f'J2={round(Js2, 2)}')

    ax[0].set_xlim(-5,2)
    ax[0].set_ylim(-5,5)

    ax[2].set_xlabel('$Re$')
    ax[0].set_ylabel('$Im$')
    ax[1].set_ylabel('$Im$')
    ax[2].set_ylabel('$Im$')

    ax[0].plot([0, 0], [-5, 5], c='grey')
    ax[0].plot([-5, 2], [0, 0], c='grey')
    ax[1].plot([0, 0], [-5, 5], c='grey')
    ax[1].plot([-5, 2], [0, 0], c='grey')
    ax[2].plot([0, 0], [-5, 5], c='grey')
    ax[2].plot([-5, 2], [0, 0], c='grey')

    J = [-2.5, Js2]
    re, ve, se = get_equilibria(g, J)
    for k in range(len(re)):
        ax[k].set_title(f'r={re[k]}')
        w = get_eigenvalues(g, J, re[k], ve[k], se[k])
        for el in w:
            ax[k].scatter(np.real(el), np.imag(el), c=colors[k], zorder=10)
        if len(re) == 1 : 
            ax[1].set_title('blank', color='white')
            ax[2].set_title('blank', color='white')
    plt.tight_layout()
    plt.savefig(f'{i}.png', dpi=400)
    plt.close()
    i +=1

