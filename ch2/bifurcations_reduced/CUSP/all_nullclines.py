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

fig, ax = plt.subplots(2, 3, figsize=(12+.25,8), gridspec_kw={'height_ratios': [1,1], 'width_ratios':[1,1,1]}) #sharex='row', sharey='row',
colors = plt.cm.plasma(np.linspace(0,0.75,5))


def sgn(x):
    return 1 if x > 0 else -1

# Parameters
I = 0
eta = -1

# Range of r and v
r_min, r_max = 0, 3
v_min, v_max = -2, 2
r = np.linspace(0,2,10**6)

''' SADDLE NODE BOUNDARIES + CUSP '''
g_m = 2/r - np.sqrt(1/(r**2) - 4*r**2 - 4*eta)
J_p = (1/(2*r**3))*(4*r**4 - 1 + np.sqrt(1 - 4*r**4 - 4*eta*r**2))
ax[0,0].plot(g_m, J_p, color='k', label='saddle-node')

g_p = 2/r + np.sqrt(1/(r**2) - 4*r**2 - 4*eta)
J_m = (1/(2*r**3))*(4*r**4 - 1 - np.sqrt(1 - 4*r**4 - 4*eta*r**2))
ax[0,0].plot(g_p, J_m, color='k')

rz = math.sqrt(-sgn(eta)* 1/2 * math.sqrt(-1+ (1+eta**2)**(1/3)) \
    + 1/2 * math.sqrt( - 2 - (1+eta**2)**(1/3) + sgn(eta)* 2*eta/(math.sqrt(-1+(1+eta**2)**(1/3)))  ) )
g_cusp, J_cusp = 2/rz - np.sqrt(1/(rz**2) - 4*rz**2 - 4*eta), (1/(2*rz**3))*(4*rz**4 - 1 + np.sqrt(1 - 4*rz**4 - 4*eta*rz**2))
ax[0,0].scatter(g_cusp, J_cusp, c=col[0], zorder=10, s=60, label='cusp' )
ax[0,0].legend(loc='lower left')

ax[0,0].set_xlabel('$\\tilde{g}$')
ax[0,0].set_ylabel('$\\tilde{J}$')
ax[0,0].set_xlim(0, 4)
ax[0,0].set_ylim(-1, 4)

ax[0,0].annotate('$T_1$', (2,3), xytext=(2,3.1))
ax[0,0].annotate('$T_2$', (2,0), xytext=(2,1.1))
ax[0,0].annotate('$(I)$', (2.5,2), xytext=(2.5,2))
ax[0,0].annotate('$(II)$', (2,0), xytext=(1.5,.75))


def dF(y1, y2, g, J): #y1 = r, y2 = v
    return 1 + 2*y1*y2 - g*y1, y2**2 + J*y1 - y1**2 + eta + I

def plot(k1, k2, g, J):
    # Phase portrait
    R, V = np.meshgrid(np.linspace(r_min, r_max, 100), np.linspace(v_min, v_max, 100))
    u, v = np.zeros_like(R), np.zeros_like(V)
    NI, NJ = R.shape

    for i in range(NI):
        for j in range(NJ):
            x, y = R[i, j], V[i, j]
            fp = dF(x, y, g, J)
            u[i,j] = fp[0]
            v[i,j] = fp[1]

    ax[k1,k2].set_xlim(r_min, r_max)
    ax[k1,k2].set_ylim(v_min, v_max)

    # Flow of dynamics
    ax[k1,k2].streamplot(R, V, u, v,  linewidth=1, color='grey', density = 1.2)

    # Plotting the nullclines
    r = np.linspace(r_min, r_max, 10**5)

    ax[k1,k2].plot(r, g/2 - 1/(2*r), linewidth=3, color=col[0], label='$\\tilde{r}$-nullcline')
    ax[k1,k2].plot(r, np.sqrt(-eta+r**2-J*r -I), linewidth=3, color=col[1], label='$\\tilde{v}$-nullcline')
    ax[k1,k2].plot(r, -np.sqrt(-eta+r**2-J*r -I), linewidth=3, color=col[1])

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

    print(k1, k2, ' : ', vs, rs)

    if k1 == 0 and k2 == 1:
        w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
        print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)
        w = np.array([1/2*(4*vs[2] -g + cmath.sqrt(g**2 + 8*rs[2]*(J-2*rs[2]))), 1/2*(4*vs[2] -g - cmath.sqrt(g**2 + 8*rs[2]*(J-2*rs[2])))])
        print(f'Equilibria : ({rs[2]}, {vs[2]})\n', 'Eigenvalues:', w)

        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)
        ax[k1,k2].scatter(rs[2], vs[2], c='white', edgecolor="black", marker=MarkerStyle("o", fillstyle="right"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[2], vs[2], c='black', edgecolor="black", marker=MarkerStyle("o", fillstyle="left"),  s=80, zorder=10)

    if k1 == 0 and k2 == 2:
        w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
        print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)
        w = np.array([1/2*(4*vs[2] -g + cmath.sqrt(g**2 + 8*rs[2]*(J-2*rs[2]))), 1/2*(4*vs[2] -g - cmath.sqrt(g**2 + 8*rs[2]*(J-2*rs[2])))])
        print(f'Equilibria : ({rs[2]}, {vs[2]})\n', 'Eigenvalues:', w)

        ax[k1,k2].scatter(rs[2], vs[2], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)
        ax[k1,k2].scatter(rs[0], vs[0], c='white', edgecolor="black", marker=MarkerStyle("o", fillstyle="right"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o", fillstyle="left"),  s=80, zorder=10)

    if k1 == 1 and k2 == 0:
        #w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
        #print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)

        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)
    
    if k1 == 1 and k2 == 1:
        w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
        print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)

        w = np.array([1/2*(4*vs[1] -g + cmath.sqrt(g**2 + 8*rs[1]*(J-2*rs[1]))), 1/2*(4*vs[1] -g - cmath.sqrt(g**2 + 8*rs[1]*(J-2*rs[1])))])
        print(f'Equilibria : ({rs[1]}, {vs[1]})\n', 'Eigenvalues:', w)

        w = np.array([1/2*(4*vs[2] -g + cmath.sqrt(g**2 + 8*rs[2]*(J-2*rs[2]))), 1/2*(4*vs[2] -g - cmath.sqrt(g**2 + 8*rs[2]*(J-2*rs[2])))])
        print(f'Equilibria : ({rs[2]}, {vs[2]})\n', 'Eigenvalues:', w)

        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)
        ax[k1,k2].scatter(rs[1], vs[1], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)
        ax[k1,k2].scatter(rs[2], vs[2], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)

    if k1 == 1 and k2 == 2:
        w = np.array([1/2*(4*vs[0] -g + cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0]))), 1/2*(4*vs[0] -g - cmath.sqrt(g**2 + 8*rs[0]*(J-2*rs[0])))])
        print(f'Equilibria : ({rs[0]}, {vs[0]})\n', 'Eigenvalues:', w)

        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="black", marker=MarkerStyle("o"),  s=60, zorder=10)

plot(0, 1, 0.9, 2.03182) #T1
plot(0, 2, 0.9, 1.996445) #T2
plot(1, 0, g_cusp, J_cusp) #CUSP
plot(1, 1, 1.2, 2.06) #INSIDE
plot(1, 2, 1, 1.75) #OUTSIDE

ax[0,0].set_title('Cusp bifurcation', pad=15)
ax[0,1].set_title('Phase space on curve $T_1$', pad=15)
ax[0,2].set_title('Phase space on curve $T_2$', pad=15) 
ax[1,0].set_title('Phase space at cusp point', pad=15)
ax[1,1].set_title('Phase space in area $(I)$', pad=15)
ax[1,2].set_title('Phase space in area $(II)$', pad=15)

ax[0,0].legend(loc='lower left', facecolor='white', framealpha=1)
ax[0,1].legend(loc='lower right', facecolor='white', framealpha=1)

ax[1,0].set_xlabel('$\\tilde{r}$')
ax[1,1].set_xlabel('$\\tilde{r}$')
ax[1,2].set_xlabel('$\\tilde{r}$')

ax[0,1].set_ylabel('$\\tilde{v}$', labelpad=-5)
ax[1,0].set_ylabel('$\\tilde{v}$', labelpad=-5)

ax[0,1].set_xticklabels([])
ax[0,2].set_xticklabels([])

ax[0,2].set_yticklabels([])
ax[1,1].set_yticklabels([])
ax[1,2].set_yticklabels([])

plt.tight_layout()
plt.subplots_adjust(wspace=.25)
plt.savefig('CUSP_phases.png', dpi=600)
plt.show()