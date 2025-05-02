import matplotlib.pyplot as plt
import math as math
import numpy as np
import auto

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 11#9
plt.rcParams['legend.fontsize'] = 11#7
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 11#9
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

'''
THIS IS THE REDUCED CASE

'''

lw = 2

def sgn(x):
    return 1 if x > 0 else -1

c = ['#e06666', '#f6b26b', '#93c47d', '#6fa8dc', '#8e7cc3']
plt.figure(figsize=(6, 4))

''' FOR eta_mean = -2 '''
r = np.linspace(10**(-6),2,10**5)
eta = -2

''' SADDLE NODE BOUNDARIES '''
g_m = 2/r - np.sqrt(1/(r**2) - 4*r**2 - 4*eta)
J_p = (1/(2*r**3))*(4*r**4 - 1 + np.sqrt(1 - 4*r**4 - 4*eta*r**2))
plt.plot(g_m, J_p, color='k', label='saddle node', linewidth=lw)

g_p = 2/r + np.sqrt(1/(r**2) - 4*r**2 - 4*eta)
J_m = (1/(2*r**3))*(4*r**4 - 1 - np.sqrt(1 - 4*r**4 - 4*eta*r**2))
plt.plot(g_p, J_m, color='k', linewidth=lw)

v_p = g_p/2 + 1/(2*r)


''' HOPF BIRFURCATION BOUNDARIES '''
g = np.linspace(0, math.sqrt(8*(eta + math.sqrt(eta**2 + 1))), 1000)
J = -(g**3)/32 - eta*g/2 + 2/g
plt.plot(g, J, color=c[0], label='hopf', linewidth=lw)

''' HOMOCLINIC '''
hom = auto.loadbd('hom')
plt.plot(hom['G'][hom['J']<4.421], hom['J'][hom['J']<4.421], color=c[3], label='homoclinic', linewidth=lw)


''' CODIM 2 POINTS '''
''' TAKENS-BOGDANOV BIRFURCATION POINT '''
g_TB = math.sqrt(8*eta + 8*math.sqrt(eta**2 + 1))
J_TB = -(g_TB**3)/32 - eta*g_TB/2 + 2/g_TB
plt.scatter(g_TB, J_TB, zorder=10, color=c[4], s=60, label='TB')

''' CUSP '''
rz = math.sqrt(-sgn(eta)* 1/2 * math.sqrt(-1+ (1+eta**2)**(1/3)) \
    + 1/2 * math.sqrt( - 2 - (1+eta**2)**(1/3) + sgn(eta)* 2*eta/(math.sqrt(-1+(1+eta**2)**(1/3)))  ) )
g_cusp, J_cusp = 2/rz - np.sqrt(1/(rz**2) - 4*rz**2 - 4*eta), (1/(2*rz**3))*(4*rz**4 - 1 + np.sqrt(1 - 4*rz**4 - 4*eta*rz**2))
plt.scatter(g_cusp, J_cusp, c=c[1], zorder=10, s=60, label='cusp' )

''' SNSL '''
plt.scatter(1.25, 4.421, color=c[2], zorder=20, s=60, label='SNSL')

''' PLOT TWEAKS '''
plt.xlim(-.5, 2)
plt.ylim(1.5, 6)
plt.xlabel('$\\tilde{g}$', size=14)
plt.ylabel('$\\tilde{J}$', size=14)
#plt.title('Homoclinic bifurcation', size=16)
plt.legend(loc='upper left', prop={'size':10})

# Annotations for areas
plt.annotate('$(I)$', (1,3.5), xytext=(.88,3.5), fontsize=16)
plt.annotate('$(II)$', (2,3.5), xytext=(1.4,3.5), fontsize=16)

''' GENERAL PLOT SETTINGS '''
plt.tight_layout()
plt.savefig(f'TB_bif.png', dpi=300)
plt.show()
