import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import cmath as cmath
import numpy as np


SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=24)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

input_file = open('input_current.dat', 'r')
current = []
for el in input_file.readline()[:-2].split(',') :
    current.append(float(el))
current = np.array(current)

# Fixed parameters
dt = 10**(-3)
neurons = 10**4

vr, vth = -100, 1000
a = abs(vth/vr)
g, J = .5, -10
tau, tau_d = 1, 1
eta_mean = 1
deltas = [0.3]# [0.1, 0.2, 0.3, 0.4]
max_time = 4000

def euler(I, dt=10**(-3)):
    v, r, s = [0.], [0.], [0.]
    #ax.scatter3D(r,v,s)
    vs = [v[0] + tau*math.log(a)*r[0]]
    current = I
    for i in range(1, int(max_time/dt)) :
        r.append(r[i-1] + dt*(delta/(tau*math.pi) + 2*r[i-1]*v[i-1] - g*r[i-1] - 2*tau*math.log(a)*r[i-1]**2)/tau)
        v.append(v[i-1] + dt*(pow(v[i-1],2) + eta_mean - pow(math.pi*tau*r[i-1],2) - pow(math.log(a)*tau*r[i-1],2) + J*tau*s[i-1] + delta*math.log(a)/math.pi)/tau)
        s.append(s[i-1] + dt*(-s[i-1] + r[i-1])/tau_d)
        #v.append(vs[i-1] + 0*tau*math.log(a)*r[i-1] )
    return np.array(v), np.array(r), np.array(s)

for delta in deltas :
    # Analytical solution
    v_sol, r_sol, s_sol = euler(I=current)

    w = math.pi*tau*r_sol + 1j*v_sol
    z = 1-w/(1+w)

    #ax.plot3D(np.real(z), np.imag(z), s_sol, 'gray')
    ax.plot3D(r_sol, v_sol, s_sol, linewidth=2.5, color='gray')
    #ax.scatter3D(0.15837235198398597, 0.19851756111826546, 0.15837235198398597)

# plot the surface r = s
#r = np.linspace(0.01, 2, 1000)
#v = np.linspace(-5, 5, 1000)

#vv,rr = np.meshgrid(v, r)
#ss = rr
#ax = plt.subplot(projection='3d')
#ax.plot_surface(vv, rr, ss, alpha=.5)

# Labelling
#ax.set_xlabel('re(z)')
ax.set_xlabel(r'$r$', labelpad=15)
#ax.set_ylabel('im(z)')
ax.set_ylabel(r'$v_{avg}$', labelpad=20)
ax.set_zlabel(r'$s$', labelpad=15)

#ax.set_zticklabels(ax.get_zticks(), fontdict={'ha':'left'})

#ax.set_xlim(0.01, 2)
#ax.set_ylim(-5,5)
#ax.set_zlim(0, 4)

#plt.tight_layout()
plt.savefig('3D_focus.png', dpi=600)

plt.show()
