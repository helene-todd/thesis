import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import random as random
import math as math
import numpy as np
import os

plt.rcParams['axes.xmargin'] = 0

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=20)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax = plt.subplots(1, 2, figsize=(10,3.5))

lnw=2

#np.random.seed(42)

# Parameters
h = 10**(-4)
neurons = 2

vr, vp = -100, 100
a = abs(vp/vr)
g, J = 2, 0
tau, tau_d = 1, 1
eta_mean = -3
delta = .3
I0, I1 = 10, 0

t_final, t_init = 16+2, 0

current_start0, current_stop0 = 6+2, 7+2
current_start1, current_stop1 = 6+2, 7+2
v0 = -1.5

steps = int((t_final - t_init)/h)
#refract_steps = 1 #int(tau/(2*h*vp)) 

# Initialize eta vector
eta = [0 for n in range(neurons)]
for n in range(neurons):
    eta[n] = delta*math.tan(math.pi*(random.random()-0.5))+eta_mean

# Initialize input current vector
current = [[0 for i in range(steps+1)] for n in range(neurons)]
for i in range(len(current[0])):
    if i >= int(current_start0/h) and i <= int(current_stop0/h) :
        current[0][i] = I0

for i in range(len(current[1])):
    if i >= int(current_start1/h) and i <= int(current_stop1/h) :
        current[1][i] = I1

        
""" NO SPIKELET ATTENUATION """
# Initialize mean membrane potential vector
v_avg = [0 for s in range(steps+1)]
v_avg[0] = v0

# Initialize membrane potential matrix
v = [[0 for n in range(neurons)], [0 for n in range(neurons)]]
v[0][0], v[0][1] = 2, -2
#for n in range(neurons):
    #v[0][n] = np.random.normal(v0, 1)
init_v0 = [v[0][0], v[0][1]]

# Initialize kuramoto order parameter vector
z = [1 for z in range(steps+1)]

# Initialize spike, synaptic activation and synaptic gating vectors
spike_times = [0 for n in range(neurons)]
r = [[0 for s in range(steps+1)] for n in range(neurons)]
s = [[0 for s in range(steps+1)] for n in range(neurons)]

# Saving values
voltages = [[] for n in range(neurons)]

# Loop
for i in tqdm(range(1, steps)):
    for n in range(neurons):
        s[n][i] = s[n][i-1] + h*(-s[n][i-1])/tau_d + r[n][i-1]
        
        if(spike_times[n] == 0 and v[0][n] >= vp):
            spike_times[n] = i
            v[1][n] = vr
            r[n][i] += 1 

        elif (spike_times[n] == 0):
            v[1][n] = v[0][n] + h*(pow(v[0][n], 2) + eta[n] + g*(v[0][1-n]-v[0][n]) + J*tau*s[1-n][i-1] + current[n][i-1])/tau

        else :
            spike_times[n] = 0

        v[0][n] = v[1][n]
        
        # Save values
        voltages[n].append(v[1][n])

times = [float(h*k) for k in range(1, steps)]
v1 = np.array(voltages[0])
v2 = np.array(voltages[1])

ax[0].plot(times, v1, color='black', alpha=0.5, label='$v_{th}=$'+f'${vp}$',linewidth=lnw, zorder=0)
ax[1].plot(times, v2, color='black', label='$v_{th}=$'+f'${vp}$',linewidth=lnw, alpha=.5, zorder=0)

""" WITH SPIKELET ATTENUATION """
# Initialize mean membrane potential vector
v_avg = [0 for s in range(steps+1)]
v_avg[0] = v0

# Initialize membrane potential matrix
v = [[0 for n in range(neurons)], [0 for n in range(neurons)]]
for n in range(neurons):
    v[0][n] = init_v0[n]

# Initialize kuramoto order parameter vector
z = [1 for z in range(steps+1)]

# Initialize spike, synaptic activation and synaptic gating vectors
spike_times = [0 for n in range(neurons)]
r = [[0 for s in range(steps+1)] for n in range(neurons)]
s = [[0 for s in range(steps+1)] for n in range(neurons)]

# Saving values
voltages = [[] for n in range(neurons)]

# Modification of threshold values
vp = 20
vr = -100

# Loop
for i in tqdm(range(1, steps)):
    for n in range(neurons):
        s[n][i] = s[n][i-1] + h*(-s[n][i-1])/tau_d + r[n][i-1]
        
        '''
        if(spike_times[n] == 0 and v[0][n] >= vp):
            spike_times[n] = i
        elif (spike_times[n] == 0):
            v[1][n] = v[0][n] + h*(pow(v[0][n], 2) + eta[n] + g*(v[0][1-n]-v[0][n]) + J*tau*s[1-n][i-1] + current[n][i-1])/tau

        elif (i <= spike_times[n] + refract_steps):
            v[1][n] = vp
            if (i <= spike_times[n] + 1):
                r[n][i] += 1 
        elif (i > spike_times[n] + refract_steps and i <= spike_times[n] + 2*refract_steps):
            v[1][n] = vr
        else :
            spike_times[n] = 0
        '''

        if(spike_times[n] == 0 and v[0][n] >= vp):
            spike_times[n] = i
            v[1][n] = vr
            r[n][i] += 1 

        elif (spike_times[n] == 0):
            v[1][n] = v[0][n] + h*(pow(v[0][n], 2) + eta[n] + g*(v[0][1-n]-v[0][n]) + J*tau*s[1-n][i-1] + current[n][i-1])/tau

        else :
            spike_times[n] = 0

        v[0][n] = v[1][n]
        
        # Save values
        voltages[n].append(v[1][n])
    
times = [float(h*k) for k in range(1, steps)]
v1 = np.array(voltages[0])
v2 = np.array(voltages[1])

ax[0].plot(times, v1, color='#aa3863', alpha=1, label='$v_{th}=20$',linewidth=lnw, zorder=-20)
ax[1].plot(times, v2, color='#3b7c86', alpha=1, label='$v_{th}=20$',linewidth=lnw, zorder=-20)

ax[0].set_xlabel('time', size=14)
ax[1].set_xlabel('time', size=14)

#ax[0].set_ylabel('voltage $v_{k}, k \in \{1,2\}$', size=14)
ax[0].set_ylabel('voltage', size=14)

ax[0].set_title('Presynaptic spike', size=14)
ax[1].set_title('Post-synaptic spikelet', size=14)
ax[0].legend(loc='upper right', fontsize=12)
ax[1].legend(loc='upper right', fontsize=12)

ax[0].set_xlim(2,18)
ax[0].set_xticks([2, 7, 12, 17], [0, 5, 10, 15])
ax[1].set_xlim(2,18)
ax[1].set_xticks([2, 7, 12, 17], [0, 5, 10, 15])

#ax[0].set_ylim(-5,5)
#ax[1].set_ylim(-5,5)

plt.subplots_adjust(wspace=10)
plt.tight_layout()
plt.savefig('spikelet_a.png', dpi=300)
plt.show()
