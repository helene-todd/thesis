import numpy as np
import math as math
import cmath as cmath
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

plt.figure(figsize=(12,6))

plt.rcParams['axes.xmargin'] = 0

# Parameters
I = 0
eta = -2
J = -5

# Range of r and v
v_min, v_max = -4, 0

def dF(y1, y2, g): #y1 = r, y2 = v
    return 1 + 2*y1*y2 - g*y1, y2**2 + J*y1 - y1**2 + eta + I

def equilibria(g):
    """ COMPUTING EQUILIBRIA """
    a = -1
    b = J
    c = g**2/4 + eta + I
    d = -g/2
    e = 1/4 

    coeff = [a, b, c, d, e]

    rs = []
    for sol in np.roots(coeff) :
        if np.imag(sol) == 0 and np.real(sol) >= 0 :
            rs.append(np.real(sol))
    rs = np.array(rs)
    vs = g/2 - 1/(2*rs)
    w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
    print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)

    return w.imag, w.real


gs = np.linspace(0, 5, 1000)
complex_part_p = []
complex_part_m = []
real_part_p = []
real_part_m = []
for g in gs:
    im, re = equilibria(g)
    if len(im) > 0:
        complex_part_p.append(im[0])
        complex_part_m.append(im[1])
    else: 
        complex_part_p.append(im[0])
        complex_part_m.append(im[0])

    if len(re) > 0:
        real_part_p.append(re[0])
        real_part_m.append(re[1])
    else: 
        real_part_p.append(re[0])
        real_part_m.append(re[0])

plt.plot(gs, complex_part_m, c='#40C1AC', linewidth=2, label='$\Im(\lambda_\pm)$')
plt.plot(gs, complex_part_p, c='#40C1AC',  linewidth=2)

plt.plot(gs, real_part_m, c='k', linewidth=2, label='$\Re(\lambda_\pm)$')
plt.plot(gs, real_part_p, c='k', linewidth=2)

plt.xlabel(r'electrical coupling $\tilde{g}$')
#plt.ylabel('$\Im$ and $\Re$ of eigenvalues $\lambda_\pm$')

plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('eigenvalues.png', dpi=300)

plt.show()