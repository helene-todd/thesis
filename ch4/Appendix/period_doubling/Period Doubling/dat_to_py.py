import matplotlib.pyplot as plt
import math as math
import numpy as np

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 12#9
plt.rcParams['legend.fontsize'] = 12#7
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 12#9
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)


lw = 2
c = ['#e06666', '#f6b26b', '#93c47d', '#6fa8dc', '#8e7cc3']

fig, ax = plt.subplots(5, 2, figsize=(12,9), width_ratios=[5, 1], sharex='col', sharey='row' )

files_r = ['J2=-1.56.dat', 'J2=-1.62.dat', 'J2=-1.64.dat', 'J2=-1.7.dat', 'J2=-1.72.dat'] 
files_rv = ['J2=-1.56rv.dat', 'J2=-1.62rv.dat', 'J2=-1.64rv.dat', 'J2=-1.7rv.dat', 'J2=-1.72rv.dat'] 
labels= ['$J_2=-1.56$', '$J_2=-1.62$', '$J_2=-1.64$', '$J_2=-1.7$', '$J_2=-1.72$']
k=0
for file in files_r:
    t, r = [], []
    f = open(file, "r")
    for line in f:
        line = line.rstrip(' \n')
        line = line.split(' ')
        t.append(float(line[0]))
        r.append(float(line[1]))

    ax[k, 0].plot(t, r, color=c[k], linewidth=lw) #label=labels[k]
    k+=1

k=0
for file in files_rv:
    r, v = [], []
    f = open(file, "r")
    for line in f:
        line = line.rstrip(' \n')
        line = line.split(' ')
        r.append(float(line[0]))
        v.append(float(line[1]))

    ax[k, 1].plot(v, r, color=c[k], linewidth=.5)
    k+=1

''' PLOT TWEAKS '''
for k in range(5):
    ax[k, 0].set_xlim(0,160)
    ax[k, 0].set_ylim(0.02, 0.26)
    ax[k, 0].set_ylabel('$r_1$')
    #ax[k, 0].legend(loc='upper left', fontsize=10)

    ax[k, 0].spines[['right', 'top']].set_visible(False)
    ax[k, 1].spines[['right', 'top']].set_visible(False)

ax[4, 0].set_xlabel('time')
ax[4, 1].set_xlabel('$v_1$')
ax[4, 1].set_xlim(-2,0)

ax[0,0].text(2, 0.215, '$1:1$ periodic')
ax[1,0].text(2, 0.215, '$2:1$ periodic')
ax[2,0].text(2, 0.215, '$4:1$ periodic')
ax[3,0].text(2, 0.215, 'chaotic')
ax[4,0].text(2, 0.215, 'collapse')

''' GENERAL PLOT SETTINGS '''
plt.tight_layout()
plt.savefig('PD_cascades.png', dpi=300)
plt.show()
