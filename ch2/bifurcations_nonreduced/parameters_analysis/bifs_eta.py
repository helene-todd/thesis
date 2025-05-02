import matplotlib.pyplot as plt
import math as math
import numpy as np

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

'''
THIS IS THE NON REDUCED CASE

'''

def sgn(x):
    return 1 if x > 0 else -1

c = ['#e06666', '#f6b26b', '#93c47d', '#6fa8dc', '#8e7cc3']
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, sharex=True)

''' PARAMETERS '''
tau = 1
delta = 1
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

axs[0].plot(g_p, J_m, color=c[1], label='focus node')
axs[0].plot(g_m, J_p, color=c[1])

''' SADDLE NODE BOUNDARIES '''
g_m = 2*delta/(math.pi*tau*r) - np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_p = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
axs[0].plot(g_m, J_p, color=c[0], label='saddle node')

g_p = 2*delta/(math.pi*tau*r) + np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_m = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) - \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
axs[0].plot(g_p, J_m, color=c[0])

''' HOPF BIRFURCATION BOUNDARIES '''
g_TB = math.sqrt(8*(eta+I) + 8*math.sqrt((eta+I)**2 + delta**2))
g = np.linspace(0, g_TB, 1000)
J = -(math.pi*g**3)/(32*delta) - (eta+I)*g*math.pi/(2*delta) + (2*delta*math.pi)/g
axs[0].plot(g, J, color=c[2], label='hopf')

''' TAKENS-BOGDANOV BIRFURCATION POINT '''
J_TB = -(math.pi*g_TB**3)/(32*delta) - (eta+I)*math.pi*g_TB/(2*delta) + 2*delta*math.pi/g_TB
axs[0].scatter(g_TB, J_TB, zorder=10, color=c[3], label='TB')

''' CUSP '''
etaz = eta/delta
rz = math.sqrt(-sgn(etaz)* 1/2 * math.sqrt(-1+ (1+etaz**2)**(1/3)) \
    + 1/2 * math.sqrt( -2 -(1+etaz**2)**(1/3) + sgn(etaz)* 2*etaz/(math.sqrt(-1+(1+etaz**2)**(1/3)))  ) )

rz = math.sqrt(delta)*rz/(tau*math.pi)

g_c = 2*delta/(math.pi*tau*rz) - np.sqrt((delta/(tau*math.pi*rz))**2 - 4*(math.pi*tau*rz)**2 - 4*eta - 4*I)
J_c = 2*tau*rz*math.pi**2 - (delta**2) / (2 * (rz**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*rz**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*rz**4 - (4*(rz**2)*(eta+I))/((tau**2)*math.pi**2))
axs[0].scatter(g_c, J_c, color=c[0], zorder=20, label='cusp')

''' PLOT TWEAKS '''
axs[0].set_ylim(-15, 15)
axs[0].set_xlim(-2, 8)
axs[0].set_xlabel('$g$')
axs[0].set_ylabel('$J$')
axs[0].set_title('$\\bar{\eta}=$'+f'{eta}')


''' FOR eta_mean = 0 '''
r = np.linspace(10**(-6),2,10**5)
eta = 0

''' STABLE / FOCUS NODE BOUNDARIES '''
g_p = 2*delta/(math.pi*tau*r) + np.sqrt(2*(delta**2)/((math.pi*tau*r)**2) - 8*(eta + (math.pi*tau*r)**2 + I))
g_m = 2*delta/(math.pi*tau*r) - np.sqrt(2*(delta**2)/((math.pi*tau*r)**2) - 8*(eta + (math.pi*tau*r)**2 + I))
J_p = (eta+I)/(tau*r) + 3*(math.pi**2)*tau*r - (3/4) * (delta**2)/((math.pi**2)*(tau*r)**3) + \
(1/2) * (delta/(math.pi*(tau*r)**2)) * np.sqrt(2*(delta**2)/(math.pi*tau*r)**2 - 8*(eta + (math.pi*tau*r)**2 + I) )
J_m = (eta+I)/(tau*r) + 3*(math.pi**2)*tau*r - (3/4) * (delta**2)/((math.pi**2)*(tau*r)**3) - \
(1/2) * (delta/(math.pi*(tau*r)**2)) * np.sqrt(2*(delta**2)/(math.pi*tau*r)**2 - 8*(eta + (math.pi*tau*r)**2 + I) )

axs[1].plot(g_p, J_m, color=c[1], label='focus node')
axs[1].plot(g_m, J_p, color=c[1])

''' SADDLE NODE BOUNDARIES '''
g_m = 2*delta/(math.pi*tau*r) - np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_p = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
axs[1].plot(g_m, J_p, color=c[0], label='saddle node')

g_p = 2*delta/(math.pi*tau*r) + np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_m = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) - \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
axs[1].plot(g_p, J_m, color=c[0])

''' HOPF BIRFURCATION BOUNDARIES '''
g_TB = math.sqrt(8*(eta+I) + 8*math.sqrt((eta+I)**2 + delta**2))
g = np.linspace(0, g_TB, 1000)
J = -(math.pi*g**3)/(32*delta) - (eta+I)*g*math.pi/(2*delta) + (2*delta*math.pi)/g
axs[1].plot(g, J, color=c[2], label='hopf')

''' TAKENS-BOGDANOV BIRFURCATION POINT '''
J_TB = -(math.pi*g_TB**3)/(32*delta) - (eta+I)*math.pi*g_TB/(2*delta) + 2*delta*math.pi/g_TB
axs[1].scatter(g_TB, J_TB, zorder=10, color=c[3], label='TB')

''' CUSP CODIM2 BIFURCATION '''
rc = math.sqrt(delta)*(-3/4 + math.sqrt(3)/2)**(1/4) / (math.pi*tau)
g_c = 2*delta/(math.pi*tau*rc) - np.sqrt((delta/(tau*math.pi*rc))**2 - 4*(math.pi*tau*rc)**2 - 4*eta - 4*I)
J_c = 2*tau*rc*math.pi**2 - (delta**2) / (2 * (rc**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*rc**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*rc**4 - (4*(rc**2)*(eta+I))/((tau**2)*math.pi**2))
axs[1].scatter(g_c, J_c, zorder=10, color=c[0], label='cusp')

''' PLOT TWEAKS '''
axs[1].set_ylim(-15, 15)
axs[1].set_xlim(-2, 8)
axs[1].set_xlabel('$g$')
axs[1].set_title('$\\bar{\eta}=$'+f'{eta}')


''' FOR eta_mean = 1 '''
r = np.linspace(10**(-6),2,10**5)
eta = 1

''' STABLE / FOCUS NODE BOUNDARIES '''
g_p = 2*delta/(math.pi*tau*r) + np.sqrt(2*(delta**2)/((math.pi*tau*r)**2) - 8*(eta + (math.pi*tau*r)**2 + I))
g_m = 2*delta/(math.pi*tau*r) - np.sqrt(2*(delta**2)/((math.pi*tau*r)**2) - 8*(eta + (math.pi*tau*r)**2 + I))
J_p = (eta+I)/(tau*r) + 3*(math.pi**2)*tau*r - (3/4) * (delta**2)/((math.pi**2)*(tau*r)**3) + \
(1/2) * (delta/(math.pi*(tau*r)**2)) * np.sqrt(2*(delta**2)/(math.pi*tau*r)**2 - 8*(eta + (math.pi*tau*r)**2 + I) )
J_m = (eta+I)/(tau*r) + 3*(math.pi**2)*tau*r - (3/4) * (delta**2)/((math.pi**2)*(tau*r)**3) - \
(1/2) * (delta/(math.pi*(tau*r)**2)) * np.sqrt(2*(delta**2)/(math.pi*tau*r)**2 - 8*(eta + (math.pi*tau*r)**2 + I) )

axs[2].plot(g_p, J_m, color=c[1], label='focus node')
axs[2].plot(g_m, J_p, color=c[1])

''' SADDLE NODE BOUNDARIES '''
g_m = 2*delta/(math.pi*tau*r) - np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_p = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
axs[2].plot(g_m, J_p, color=c[0], label='saddle node')

g_p = 2*delta/(math.pi*tau*r) + np.sqrt((delta/(tau*math.pi*r))**2 - 4*(math.pi*tau*r)**2 - 4*eta - 4*I)
J_m = 2*tau*r*math.pi**2 - (delta**2) / (2 * (r**3) * (tau**3) * math.pi**2) - \
(delta/(2*tau*r**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*r**4 - (4*(r**2)*(eta+I))/((tau**2)*math.pi**2))
axs[2].plot(g_p, J_m, color=c[0])

''' HOPF BIRFURCATION BOUNDARIES '''
g_TB = math.sqrt(8*(eta+I) + 8*math.sqrt((eta+I)**2 + delta**2))
g = np.linspace(0, g_TB, 1000)
J = -(math.pi*g**3)/(32*delta) - (eta+I)*g*math.pi/(2*delta) + (2*delta*math.pi)/g
axs[2].plot(g, J, color=c[2], label='hopf')

''' TAKENS-BOGDANOV BIRFURCATION POINT '''
J_TB = -(math.pi*g_TB**3)/(32*delta) - (eta+I)*math.pi*g_TB/(2*delta) + 2*delta*math.pi/g_TB
axs[2].scatter(g_TB, J_TB, zorder=10, color=c[3], label='TB')

''' CUSP '''
etaz = eta/delta
rz = math.sqrt(-sgn(etaz)* 1/2 * math.sqrt(-1+ (1+etaz**2)**(1/3)) \
    + 1/2 * math.sqrt( -2 -(1+etaz**2)**(1/3) + sgn(etaz)* 2*etaz/(math.sqrt(-1+(1+etaz**2)**(1/3)))  ) )

rz = math.sqrt(delta)*rz/(tau*math.pi)

g_c = 2*delta/(math.pi*tau*rz) - np.sqrt((delta/(tau*math.pi*rz))**2 - 4*(math.pi*tau*rz)**2 - 4*eta - 4*I)
J_c = 2*tau*rz*math.pi**2 - (delta**2) / (2 * (rz**3) * (tau**3) * math.pi**2) + \
(delta/(2*tau*rz**3))*np.sqrt((delta**2)/((tau**4)*math.pi**4) - 4*rz**4 - (4*(rz**2)*(eta+I))/((tau**2)*math.pi**2))
axs[2].scatter(g_c, J_c, color=c[0], zorder=20, label='cusp')

''' PLOT TWEAKS '''
axs[2].set_ylim(-15, 15)
axs[2].set_xlim(-2, 8)
axs[2].set_xlabel('$g$')
axs[2].set_title('$\\bar{\eta}=$'+f'{eta}')
axs[2].legend(loc='upper right')


''' GENERAL PLOT SETTINGS '''
plt.tight_layout()
plt.savefig('bifs_eta.png', dpi=600)
plt.show()
