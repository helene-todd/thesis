import numpy as np
import math as math
import cmath as cmath
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


col = ['#aa3863', '#31688e']
colors = plt.cm.plasma(np.linspace(0,0.75,5))
diamond_color = '#cf5991'

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14#9
plt.rcParams['legend.fontsize'] = 12#7
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 14#9
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True) 

# Parameters
I = 0
eta = -2
J = 3.6

# Range of r and v
r_min, r_max = 0.001, 6.5
v_min, v_max = -2, 2.5

gray = '#241f31ff'

def dF(y1, y2, g): #y1 = r, y2 = v
    return 1 + 2*y1*y2 - g*y1, y2**2 + J*y1 - y1**2 + eta + I

def plot(k, g):
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

    ax[k].set_xlim(0, r_max)
    ax[k].set_ylim(v_min, v_max)

    # Speed of dynamics
    #speed = np.sqrt(u**2 + v**2)
    #ax[k].pcolor(R, V, np.log10(speed), cmap='viridis')

    # Flow of dynamics
    ax[k].streamplot(R, V, u, v,  linewidth=1, color='grey', density = 1.2)

    # Plotting the nullclines
    r = np.linspace(r_min, r_max, 10**5)

    ax[k].plot(r, g/2 - 1/(2*r), linewidth=3, color=col[0], label='$\\tilde{r}$-nullcline')
    ax[k].plot(r, np.sqrt(-eta+r**2-J*r -I), linewidth=3, color=col[1], label='$\\tilde{v}$-nullcline')
    ax[k].plot(r, -np.sqrt(-eta+r**2-J*r -I), linewidth=3, color=col[1])

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
    
    ax[k].scatter(rs[0], vs[0], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
    #ax[k].scatter(rs[1], vs[1], c='black', edgecolor="black", marker=MarkerStyle("o", fillstyle="right"),  s=80, zorder=10)
    ax[k].scatter(rs[1], vs[1], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
    ax[k].scatter(rs[2], vs[2], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)

    # Trajectory on phase portrait
    r0, v0 = 3, 1
    r, v = [r0], [v0]
    dt = 10**(-3)
    for i in range(60000) :
        rdot, vdot = dF(r[i], v[i],g)
        r.append( r[i] + dt*rdot )
        v.append( v[i] + dt*vdot )

    ax[k].plot(r[1:], v[1:], c=gray, linewidth=2.5)
    ax[k].scatter(r0, v0, s=50, c=colors[-1], edgecolor='black', marker='D', zorder=10)

plot(0, 1)
plot(1, 1.09)
plot(2, 1.6)

# Labeling the axes
ax[0].set_xlabel('$\\tilde{r}$')
ax[1].set_xlabel('$\\tilde{r}$')
ax[2].set_xlabel('$\\tilde{r}$')
ax[0].set_ylabel('$\\tilde{v}$')

ax[0].set_title('Phase space in area (I)')
ax[1].set_title('Phase space near homoclinic')
ax[2].set_title('Phase space in area (II)')

ax[0].legend(loc='lower right', framealpha=1)

plt.tight_layout()
plt.subplots_adjust(wspace=.25)
plt.savefig('HOM_phase_portraits.png', dpi=600)
plt.show()