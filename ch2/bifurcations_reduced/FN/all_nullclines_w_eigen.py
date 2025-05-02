import numpy as np
import math as math
import cmath as cmath
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


col = ['#aa3863', '#31688e']

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14#9
plt.rcParams['legend.fontsize'] = 12#7
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 14#9
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

fig, ax = plt.subplots(2, 3, figsize=(12+.25,8), sharey='row', gridspec_kw={'height_ratios': [1,1], 'width_ratios':[1,1,1]}) #sharex='row'
colors = plt.cm.plasma(np.linspace(0,0.75,3))

# Parameters
I = 0
eta = -2
J = -5

# Range of r and v
v_min, v_max = -4, 0

def dF(y1, y2, g): #y1 = r, y2 = v
    return 1 + 2*y1*y2 - g*y1, y2**2 + J*y1 - y1**2 + eta + I

def plot(k, g):
    # Range of r and v
    if k == 0:
        r_min, r_max = 0.001, .6
        ax[1,k].set_xlim(0, r_max)
        
    if k == 1:
        r_min, r_max = 0.001, .4
        ax[1,k].set_xlim(0, r_max)

    if k == 2:
        r_min, r_max = 0.001, .3
        ax[1,k].set_xlim(0, r_max)
        ax[1,k].set_xticks([0, 0.1, 0.2, 0.3])


    # Phase portrait
    R, V = np.meshgrid(np.linspace(r_min, r_max, 100), np.linspace(v_min, v_max, 100))
    u, v = np.zeros_like(R), np.zeros_like(V)
    NI, NJ = R.shape

    for i in range(NI):
        for j in range(NJ):
            x, y = R[i, j], V[i, j]
            fp = dF(x, y, g)
            u[i,j] = fp[0]
            v[i,j] = fp[1]
        
    ax[1,k].set_ylim(v_min, v_max)

    # Flow of dynamics
    ax[1,k].streamplot(R, V, u, v,  linewidth=1, color='grey', density = 1.2)

    # Plotting the nullclines
    r = np.linspace(r_min, r_max, 10**5)

    ax[1,k].plot(r, g/2 - 1/(2*r), linewidth=3, color=col[0], label='$\\tilde{r}$-nullcline')
    ax[1,k].plot(r, np.sqrt(-eta+r**2-J*r -I), linewidth=3, color=col[1], label='$\\tilde{v}$-nullcline')
    ax[1,k].plot(r, -np.sqrt(-eta+r**2-J*r -I), linewidth=3, color=col[1])

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

    print(k, ' : ', vs, rs)


    w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
    print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)

    ax[0,k].scatter(np.real(w[0]), np.imag(w[0]), zorder=10, s=80, c=colors[-1])
    ax[0,k].scatter(np.real(w[1]), np.imag(w[1]), zorder=10, s=80, c=colors[-1])

    ax[1,k].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o", fillstyle="right"),  s=80, zorder=10)
    ax[1,k].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o", fillstyle="left"),  s=80, zorder=10)


plot(0, 0)
plot(1, 2.65785)
plot(2, 5)

ax0min, ax0max = -10, 10
ax[0,0].plot([0, 0], [ax0min, ax0max], c='grey')
ax[0,0].plot([ax0min, ax0max], [0, 0], c='grey')
ax[0,1].plot([0, 0], [ax0min, ax0max], c='grey')
ax[0,1].plot([ax0min, ax0max], [0, 0], c='grey')
ax[0,2].plot([0, 0], [ax0min, ax0max], c='grey')
ax[0,2].plot([ax0min, ax0max], [0, 0], c='grey')

for k in range(3):
    ax[0,k].axis('off')

    ax[0,k].set_xlim(ax0min-4, ax0max+4)
    ax[0,k].set_ylim(ax0min-4, ax0max+4)

    ax[0,k].annotate("$\Re(\lambda)$", xy=(ax0max, 0), xytext=(ax0max+1, -.35), fontsize=14)
    ax[0,k].annotate("$\Im(\lambda)$", xy=(0, ax0max), xytext=(-1.5, ax0max+1.2), fontsize=14)

    ax[0,k].annotate('',
        xytext=(ax0max-1, 0),
        xy=(ax0max+1, 0),
        arrowprops=dict(arrowstyle="->", color='grey', lw=1.5, alpha=1),
        size=14
    )

    ax[0,k].annotate('',
        xytext=(0, ax0max-1),
        xy=(0, ax0max+1),
        arrowprops=dict(arrowstyle="->", color='grey', lw=1.5, alpha=1),
        size=14
    )

# Labeling the axes
ax[1,0].set_xlabel('$\\tilde{r}$')
ax[1,1].set_xlabel('$\\tilde{r}$')
ax[1,2].set_xlabel('$\\tilde{r}$')
ax[1,0].set_ylabel('$\\tilde{v}$')

# Titles
ax[0,0].set_title('Complex conjugate\neigenvalues')
ax[0,1].set_title('Negative real\neigenvalues')
ax[0,2].set_title('Negative real\neigenvalues')

ax[1,0].set_title('Phase space')
ax[1,1].set_title('Phase space')
ax[1,2].set_title('Phase space')

ax[1,0].legend(loc='lower right', facecolor='white', framealpha=1)

plt.tight_layout()
plt.subplots_adjust(wspace=.25)
plt.savefig('FN_phases.png', dpi=600)
plt.show()