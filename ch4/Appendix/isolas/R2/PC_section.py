from jax import jit, numpy as jnp
import numpy as np
import math, jax, tqdm, os, matplotlib
import matplotlib.pyplot as plt
import os, csv
from scipy.signal import find_peaks
from pathlib import Path

np.random.seed()

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14#9
plt.rcParams['legend.fontsize'] = 14#7
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.labelsize'] = 14#9
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.linewidth'] = '0.4'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rc('axes', axisbelow=True)

# Parameters
T = 5000
STEP = 1e-3

VR = -100
VP = 100
a = abs(VP/VR)

# Cross chemical coupling strength between the two populations
Jc = -8

tau_m = 1
tau_d = 1

eta_mean = 1
delta = .3

plt.figure(figsize=(12,6))

def euler(r0, v0, s0, g, J):
    v, r, s = [[v0[0]], [v0[1]]], [[r0[0]], [r0[1]]], [[s0[0]], [s0[1]]],
    for i in range(1, int(T/STEP)):
        r[0].append(r[0][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[0][i-1]*v[0][i-1] - g[0]*r[0][i-1] - 2*tau_m*math.log(a)*r[0][i-1]**2)/tau_m)
        r[1].append(r[1][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[1][i-1]*v[1][i-1] - g[1]*r[1][i-1] - 2*tau_m*math.log(a)*r[1][i-1]**2)/tau_m)
        v[0].append(v[0][i-1] + STEP*(v[0][i-1]**2 + eta_mean + J[0]*tau_m*s[0][i-1] + Jc*tau_m*s[1][i-1] - (math.log(a)**2 + math.pi**2)*(tau_m*r[0][i-1])**2 + delta*math.log(a)/math.pi )/tau_m)
        v[1].append(v[1][i-1] + STEP*(v[1][i-1]**2 + eta_mean + J[1]*tau_m*s[1][i-1] + Jc*tau_m*s[0][i-1] - (math.log(a)**2 + math.pi**2)*(tau_m*r[1][i-1])**2 + delta*math.log(a)/math.pi )/tau_m)
        s[0].append(s[0][i-1] + STEP*(-s[0][i-1] + r[0][i-1])/tau_d)
        s[1].append(s[1][i-1] + STEP*(-s[1][i-1] + r[1][i-1])/tau_d)
    return np.array(v[0]), np.array(v[1]), np.array(r[0]), np.array(r[1]), np.array(s[0]), np.array(s[1])

# Initial states in population 1, population 2
r0 = [0, 0]
v0 = [-10, 0]
s0 = [0, 0]

# Electrical coupling strength in population 1, population 2
g = [0.6, 2]

my_file = Path('periods_black.txt')

if not my_file.is_file():
    for J2 in np.arange(-2.9, -3.1, -0.001):
        # Chemical coupling strength in population 1, population 2
        J = [-2.5, J2]

        v1, v2, r1, r2, s1, s2 = euler(r0, v0, s0, g, J)

        r2 = r2[-int(800/STEP):]
        peaks, _ = find_peaks(r2)
        peaks_neg, _ = find_peaks(-r2)

        r2 = np.round(r2, 4)

        r_values_p = np.unique(r2[peaks])
        r_values_m = np.unique(r2[peaks_neg])

        if len(r2[peaks]) == 0 and len(r2[peaks_neg]) == 0:
            plt.scatter(J2, r2[-1], c='k', s=5)
            with open('periods.txt', 'a') as f:
                f.write(f'{J2} {r2[-1]}\n')
                print(f'{J2} {r2[-1]}')
        else: 
            plt.scatter(J2*np.ones(len(r_values_p)), r_values_p, c='k', s=5)
            plt.scatter(J2*np.ones(len(r_values_m)), r_values_m, c='k', s=5)
            r0, v0, s0 = [r1[-1], r2[-1]], [v1[-1], v2[-1]], [s1[-1], s2[-1]] #continuation
            with open('periods.txt', 'a') as f:
                f.write(f'{J2}')
                for el in r_values_m:
                    f.write(f' {el}')
                for el in r_values_p:
                    f.write(f' {el}')
                print(f'{J2} {r_values_m} {r_values_p}')
                f.write('\n')
else: 
    with open('periods_red.txt') as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.split(' ')
            line = list(filter(None, line))
            line = np.array(line, dtype=float)
            print(line)
            if line[0] >= -3.076 and line [0] <= -3.025: 
                plt.scatter(line[0]*np.ones(len(line[1:])), line[1:], c='r', s=10, alpha=.6, zorder=8)
            
    with open('periods_black.txt') as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.split(' ')
            line = list(filter(None, line))
            line = np.array(line, dtype=float)
            print(line)
            plt.scatter(line[0]*np.ones(len(line[1:])), line[1:], c='k', s=8)

plt.xlim(-2.9, -3.1)
plt.xlabel('$J_2$')
plt.ylabel('$r_2$')

plt.tight_layout()

plt.savefig('PC_section.png', dpi=300)
plt.show()