#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math, jax, tqdm, os, matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#sd=16, 31, 33
sd=35
np.random.seed()
print()

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=20)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Parameters
T = 120
STEP = 1e-3

N = 10**4

VR = -100
VP = 100

g = 3 
J = -math.pi 

tau_m = 10
tau_s = 0.1

REFRACT_TIME = tau_m/(VP)
selected_neurons = 10000

eta_mean = 1
delta = 1

# Initial states
r0 = 0
v0 = np.zeros(N)

def read_files(filename):
        if os.path.isfile(filename) :
            data_file = open(filename, 'r')
            data = []
            for el in data_file.readline()[:-2].split(',') :
                data.append(float(el))
            data_file.close()
        else :
            print(f'no file {filename}')
            exit()
        return np.array(data)

def retrieve_files() :
    v = read_files(f'v_avg.dat')
    r = read_files(f'r_avg.dat')

    if os.path.isfile(f'raster.dat') :
        raster_file = open(f'raster.dat', 'r')
        times_raster, raster = [], []
        for line in raster_file :
            times_raster.append(float(line.split('  ')[0]))
            raster.append(int(line.split('  ')[1].rstrip()))
    else :
        print(f'no file raster.dat')
        exit()
    return r, v, times_raster, raster

def euler():
    v, r = [np.mean(v0)], [r0]
    for i in range(1, int(T/STEP)):
        r.append(r[i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[i-1]*v[i-1] - g*r[i-1] )/tau_m)
        v.append(v[i-1] + STEP*(v[i-1]**2 + eta_mean + J*tau_m*r[i-1] -  (math.pi**2)*(tau_m*r[i-1])**2)/tau_m)
    return np.array(v), np.array(r)

''' PLOTTING '''

def plot_system(analytical=True) :
    # Generating subplots
    fig, ax = plt.subplots(1, 1, figsize=(12,3))

    # Retrieve simulation solutions
    r, v, times_raster, raster = retrieve_files()

    # Plotting voltage average
    #times = [float(STEP*k) for k in range(len(v))]
    #ax[0].plot(times, v, c='k')
    #ax[0].set_ylabel(r'$v(t)$')

    # Plotting firing rate average
    times = [float(STEP*k) for k in range(len(r))]
    ax.plot(times, r, c='k')
    ax.set_ylabel(r'$r$ ($1e$-$3$ Hz)')
    ax.set_ylim(0, 0.4)
    ax.set_xlim(0, T)
    ax.set_xlabel('time (ms)')

    # Plotting raster plot
    #ax[1].set_ylabel('neuron #')
    #ax[1].scatter(times_raster, raster, s=0.9, alpha=0.25, c='k')
    #ax[1].set_ylim(0, selected_neurons)
    #ax[1].set_yticks([0, 10000], [r'$0$', r'$1e4$'])
    #ax[1].set_xlabel('time')
    #ax[1].set_xlim(0, T)

    # Analytical solution
    if analytical :
        v_sol, r_sol = euler()
        #ax[0].plot([STEP*k for k in range(len(v_sol))], v_sol, c='r')
        ax.plot([STEP*k for k in range(len(r_sol))], r_sol, c='r', linewidth=2)

    #plt.suptitle('Network of 10e4 electrically and chemically coupled neurons')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    plt.savefig('full_network.png', dpi=600)

    plt.show()
    plt.close()

plot_system()
