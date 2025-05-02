import numpy as np
import math as math
import cmath as cmath
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.markers import MarkerStyle

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

col = ['#f39c12', '#e74c3c']

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

fig, ax = plt.subplots(2, 3, figsize=(14,7)) 

cmap = plt.get_cmap('viridis')
nviridis = truncate_colormap(cmap, 0.2, 1)
lw=2.5

colors = plt.cm.plasma(np.linspace(0,0.75,5))
diamond_color = colors[-1] #'#cf5991' #'#cf5991'

gray = '#241f31ff'

# Parameters
tau = 1
delta = 0.3
I = 0
eta= -1

def dF(y1, y2, g, J): #y1 = r, y2 = v
    return delta/(math.pi*tau**2) + 2*y1*y2/tau - g*y1/tau, (y2**2)/tau + eta/tau - tau*(math.pi**2)*y1**2 + J*y1 + I/tau

def plot(k1, k2, g, J, r_min, r_max, v_min, v_max):
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

    # Speed of dynamics
    speed = np.sqrt(u**2 + v**2)
    im = ax[k1,k2].pcolor(R, V, speed, vmin=0, cmap=nviridis)
    cbar = fig.colorbar(im)
    if k2==2:
        cbar.set_label('speed of dynamics', fontsize=14)
    
    # Flow of dynamics
    ax[k1,k2].streamplot(R, V, u, v,  linewidth=1, color='white', density = 1.2)

    # Plotting the nullclines
    r = np.linspace(r_min, r_max, 10**5)

    ax[k1, k2].plot(r, g/2 - delta/(2*tau*math.pi*r), linewidth=3, color=col[0], label='$\\tilde{r}$-nullcline')
    ax[k1, k2].plot(r, np.sqrt(-eta+(r*math.pi*tau)**2-J*tau*r -I), linewidth=3, color=col[1], label='$\\tilde{v}$-nullcline')
    ax[k1, k2].plot(r, -np.sqrt(-eta+(r*math.pi*tau)**2-J*tau*r -I), linewidth=3, color=col[1])

    ax[k1,k2].set_xlim(r_min, r_max)
    ax[k1,k2].set_ylim(v_min, v_max)
    ax[k1,k2].set_xticks(np.linspace(r_min, r_max,5))
    ax[k1,k2].set_yticks(np.linspace(v_min, v_max,5))

    """ COMPUTING EQUILIBRIA """
    a = -math.pi**2*tau**2
    b = J*tau
    c = g**2/4 + eta 
    d = -delta*g/(2*math.pi*tau)
    e = delta**2/(4*tau**2*math.pi**2)

    coeff = [a, b, c, d, e]

    rs = []
    for sol in np.roots(coeff) :
        if np.imag(sol) == 0 and np.real(sol) >= 0 :
            rs.append(np.real(sol))
    rs = np.array(rs)
    vs = -delta/(2*tau*math.pi*rs) + g/2

    if k1==0 and k2==0:
        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="white", marker=MarkerStyle("o"),  s=80, zorder=10)

        # Trajectory on phase portrait
        
        r0, v0 = 0.6, 0.3
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")
        

    if k1==0 and k2==1:
        ax[k1,k2].scatter(rs[0], vs[0], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)

        # Trajectory on phase portrait
        
        r0, v0 = 1, 2.2
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")
        

    if k1==0 and k2==2:
        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="white", marker=MarkerStyle("o"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[1], vs[1], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[2], vs[2], c='black', edgecolor="white", marker=MarkerStyle("o"),  s=80, zorder=10)

        # Trajectory on phase portrait
        
        r0, v0 = 0.4, 0.2
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")

        r0, v0 = 0.6, -1.2
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")


    if k1==1 and k2==0:
        ax[k1,k2].scatter(rs[0], vs[0], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[1], vs[1], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[2], vs[2], c='black', edgecolor="white", marker=MarkerStyle("o"),  s=80, zorder=10)

        # Trajectory on phase portrait
        
        r0, v0 = 0.5, 1.25
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")

        r0, v0 = 0.73, -2.2
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")


    if k1==1 and k2==1:
        ax[k1,k2].scatter(rs[0], vs[0], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[1], vs[1], c='white', edgecolor="black", marker=MarkerStyle("o"),  s=80, zorder=10)
        ax[k1,k2].scatter(rs[2], vs[2], c='black', edgecolor="white", marker=MarkerStyle("o"),  s=80, zorder=10)

        # Trajectory on phase portrait

        r0, v0 = 0.7, 0.55
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")

        
    if k1==1 and k2==2:
        ax[k1,k2].scatter(rs[0], vs[0], c='black', edgecolor="white", marker=MarkerStyle("o"),  s=80, zorder=10)
        
        # Trajectory on phase portrait

        r0, v0 = 0.15, -1.75
        r, v = [r0], [v0]
        dt = 10**(-3)
        for i in range(800000) :
            r.append( r[i] + dt*(delta/(math.pi*tau**2) + 2*r[i]*v[i]/tau - g*r[i]/tau) )
            v.append( v[i] + dt*((v[i]**2)/tau + eta/tau - tau*(math.pi**2)*r[i]**2 + J*r[i] + I/tau) )
        ax[k1,k2].plot(r[1:], v[1:], c=gray, lw=lw)
        ax[k1,k2].scatter(r0, v0, s=50, c=diamond_color, zorder=10, marker='D', edgecolor="black")
        

plot(0,0,0,10,0,1.4,-1.5,1.5) #1
plot(0,1,0.25,11.5,0,2.6,-3,3) #2
plot(0,2,0,7,0,1,-1.5,1.5) #3
plot(1,0,0.37,9.5,0,2,-2.5,2.5) #4
plot(1,1,0.6,9,0,2,-2.5,2.5) #5
plot(1,2,0.5,5,0,.2,-2,0) #6

# Labeling the axes
#ax[0,0].set_xlabel('$r$')
#ax[0,1].set_xlabel('$r$')
#ax[0,2].set_xlabel('$r$')
ax[0,0].set_ylabel('$v$')

ax[1,0].set_xlabel('$r$')
ax[1,1].set_xlabel('$r$')
ax[1,2].set_xlabel('$r$')
ax[1,0].set_ylabel('$v$')

ax[0,0].set_xticks([0, 0.7, 1.4], ['$0$', '$0.7$', '$1.4$'])
ax[0,1].set_xticks([0, 1.3, 2.6], ['$0$', '$1.3$', '$2.6$'])
ax[0,2].set_xticks([0, 0.5, 1], ['$0$', '$0.5$', '$1$'])
ax[1,0].set_xticks([0, 1, 2])
ax[1,1].set_xticks([0, 1, 2])
ax[1,2].set_xticks([0, 0.1, 0.2], ['$0$', '$0.1$', '$0.2$'])

ax[0,0].set_yticks([-1.5, 0, 1.5])
ax[0,1].set_yticks([-3, 0, 3])
ax[0,2].set_yticks([-1.5, 0, 1.5])
ax[1,0].set_yticks([-2.5, 0, 2.5])
ax[1,1].set_yticks([-2.5, 0, 2.5])
ax[1,2].set_yticks([-2, -1, 0])

ax[0,0].set_title('Phase space in area $(I)$', pad=20)
ax[0,1].set_title('Phase space in area $(II)$', pad=20)
ax[0,2].set_title('Phase space in area $(III)$', pad=20)
ax[1,0].set_title('Phase space in area $(IV)$', pad=20)
ax[1,1].set_title('Phase space in area $(V)$', pad=20)
ax[1,2].set_title('Phase space in area $(VI)$', pad=20)

ax[0, 0].legend(loc='upper left', framealpha=1)

plt.tight_layout()
fig.subplots_adjust(hspace=0.5, wspace=0.2)
plt.savefig('full_phases.png', dpi=600)
plt.show()