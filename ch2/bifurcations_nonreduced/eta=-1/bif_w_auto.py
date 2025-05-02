import matplotlib.pyplot as plt
import math as math
import numpy as np
import auto


plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 16#9
plt.rcParams['legend.fontsize'] = 14#7
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 16#9
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

'''
THIS IS THE NON REDUCED CASE

'''

lw = 2.2

def sgn(x):
    return 1 if x > 0 else -1

c = ['#e06666', '#f6b26b', '#93c47d', '#6fa8dc', '#8e7cc3']
plt.figure(figsize=(10, 5))

''' PARAMETERS '''
tau = 1
delta = .3
I = 0

''' FOR eta_mean = -2 '''
r = np.linspace(10**(-6),2,10**5)
eta = -1

''' STABLE / FOCUS NODE BOUNDARIES '''
g_p = 2*delta/(math.pi*tau*r) + np.sqrt(2*(delta**2)/((math.pi*tau*r)**2) - 8*(eta + (math.pi*tau*r)**2 + I))
g_m = 2*delta/(math.pi*tau*r) - np.sqrt(2*(delta**2)/((math.pi*tau*r)**2) - 8*(eta + (math.pi*tau*r)**2 + I))
J_p = (eta+I)/(tau*r) + 3*(math.pi**2)*tau*r - (3/4) * (delta**2)/((math.pi**2)*(tau*r)**3) + \
(1/2) * (delta/(math.pi*(tau*r)**2)) * np.sqrt(2*(delta**2)/(math.pi*tau*r)**2 - 8*(eta + (math.pi*tau*r)**2 + I) )
J_m = (eta+I)/(tau*r) + 3*(math.pi**2)*tau*r - (3/4) * (delta**2)/((math.pi**2)*(tau*r)**3) - \
(1/2) * (delta/(math.pi*(tau*r)**2)) * np.sqrt(2*(delta**2)/(math.pi*tau*r)**2 - 8*(eta + (math.pi*tau*r)**2 + I) )

plt.plot(g_p, J_m, color='grey', linestyle='--', linewidth=lw, label='focus node')
plt.plot(g_m, J_p, color='grey', linestyle='--', linewidth=lw)

''' SADDLE NODE BOUNDARIES '''
g_m = 2*delta/(math.pi*tau*r) - np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_p = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
plt.plot(g_m, J_p, color='k', label='saddle node', linewidth=lw)

g_p = 2*delta/(math.pi*tau*r) + np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_m = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) - \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
plt.plot(g_p, J_m, color='k', linewidth=lw)

''' HOPF BIRFURCATION BOUNDARIES '''
g_TB = math.sqrt(8*(eta+I) + 8*math.sqrt((eta+I)**2 + delta**2))
g = np.linspace(0, g_TB, 1000)
J = -(math.pi*g**3)/(32*delta) - (eta+I)*g*math.pi/(2*delta) + (2*delta*math.pi)/g
plt.plot(g, J, color=c[0], linewidth=lw, label='hopf')

''' HOMOCLINIC '''
homm = auto.loadbd('homm')
homp = auto.loadbd('homp')
plt.plot(homm['G'], homm['J'], color=c[3], label='homoclinic', linewidth=lw)
plt.plot(homp['G'], homp['J'], color=c[3], linewidth=lw)

''' TAKENS-BOGDANOV BIRFURCATION POINT '''
g_TB = math.sqrt(8*(eta+I) + 8*math.sqrt((eta+I)**2 + delta**2))
J_TB = -(math.pi*g_TB**3)/(32*delta) - (eta+I)*math.pi*g_TB/(2*delta) + 2*delta*math.pi/g_TB
plt.scatter(g_TB, J_TB, color=c[4], zorder=20, s=60, label='TB', linewidth=lw)

''' CUSP '''
etaz = eta/delta
rz = math.sqrt(-sgn(etaz)* 1/2 * math.sqrt(-1+ (1+etaz**2)**(1/3)) \
    + 1/2 * math.sqrt( -2 -(1+etaz**2)**(1/3) + sgn(etaz)* 2*etaz/(math.sqrt(-1+(1+etaz**2)**(1/3)))  ) )

rz = math.sqrt(delta)*rz/(tau*math.pi)

g_c = 2*delta/(math.pi*tau*rz) - np.sqrt((delta/(tau*math.pi*rz))**2 - 4*(math.pi*tau*rz)**2 - 4*eta - 4*I)
J_c = 2*tau*rz*math.pi**2 - (delta**2) / (2 * (rz**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*rz**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*rz**4 - (4*(rz**2)*(eta+I))/((tau**2)*math.pi**2))
plt.scatter(g_c, J_c, color=c[1], zorder=20, s=60, label='cusp')

''' SNSL '''
plt.scatter(0.6636, 13.669, color=c[2], zorder=20, s=60, label='SNSL')

''' PLOT TWEAKS '''
plt.ylim(-5, 20)
plt.xlim(-2, 2)
plt.xlabel('electrical coupling $g$')
plt.ylabel('chemical coupling $J$')
#plt.title('$\eta=$'+f'{eta}', size=16)
plt.legend(loc='upper left')

# Annotations for areas
plt.annotate('$(I)$', (-0.5,12.5), fontsize=14)
plt.annotate('$(II)$', (0.4, 16), fontsize=14)
plt.annotate('$(III)$', (-0.05, 7.15), fontsize=14)
plt.annotate('$(IV)$', (0.3, 9), fontsize=14)
plt.annotate('$(V)$', (1.2, 10), fontsize=14)
plt.annotate('$(VI)$', (0.25, 3), fontsize=14)

''' GENERAL PLOT SETTINGS '''
plt.tight_layout()
plt.savefig(f'bif_eta={eta}.png', dpi=300)
plt.show()
