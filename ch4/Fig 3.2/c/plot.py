#!/usr/bin/env python
# coding: utf-8

from jax import jit, numpy as jnp
import numpy as np
import math, jax, tqdm, os, matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

np.random.seed()

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title


# Parameters
T = 100
STEP = 1e-3

# Number of neurons in population 1, population 2
N = [10**4, 10**4]

VR = -100
VP = 100

# Electrical coupling strength in population 1, population 2
g = [0, 2]
# Chemical coupling strength in population 1, population 2
Js = [-2.5, 0]
# Cross chemical coupling strength between the two populations
Jc = -8

tau_m = 1
tau_d = 1
tau_s1, tau_s2 = 1*tau_m*1e-2, .5*tau_m*1e-2 # must be fitted to match analytical solution

eta_mean = 1
delta = .3

# Initialize input current vector
I = np.concatenate((np.zeros(int(T/(2*STEP))), 4*np.ones(int(T/(40*STEP))), np.zeros(int(39*T/(40*STEP)))), axis=0)

# Initial states in population 1, population 2
r0 = [0, 0]
v0 = [np.random.normal(0, 1, N[0]), np.random.normal(-10, 1, N[1])]
s0 = [0, 0]


''' GENERATE DIRECTORY & FILES FUNCTIONS '''

# Create subdirectory for future files
def create_directories() :
    directory_list = list()
    for root, dirs, files in os.walk(os.getcwd(), topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(name))

    folder_nb = 0

    # Create data files in new subdirectory
    folder = '0'
    subfolder1 = '0/Population1'
    subfolder2 = '0/Population2'

    if not os.path.exists(os.path.join(os.getcwd(), subfolder1)):
        os.makedirs(os.path.join(os.getcwd(), subfolder1))

    if not os.path.exists(os.path.join(os.getcwd(), subfolder2)):
        os.makedirs(os.path.join(os.getcwd(), subfolder2))

    if not os.path.exists(f'{subfolder1}/v_avg.dat'):
        os.mknod(f'{subfolder1}/v_avg_1.dat')

    if not os.path.exists(f'{subfolder2}/v_avg.dat'):
        os.mknod(f'{subfolder2}/v_avg_2.dat')

    if not os.path.exists(f'{subfolder1}/r_avg.dat'):
        os.mknod(f'{subfolder1}/r_avg.dat')

    if not os.path.exists(f'{subfolder2}/r_avg.dat'):
        os.mknod(f'{subfolder2}/r_avg.dat')

    if not os.path.exists(f'{subfolder1}/s.dat'):
        os.mknod(f'{subfolder1}/s.dat')

    if not os.path.exists(f'{subfolder2}/s.dat'):
        os.mknod(f'{subfolder2}/s.dat')

    if not os.path.exists(f'{subfolder1}/z.dat'):
        os.mknod(f'{subfolder1}/z.dat')

    if not os.path.exists(f'{subfolder2}/z.dat'):
        os.mknod(f'{subfolder2}/z.dat')

    if not os.path.exists(f'{folder}/input_current.dat'):
        os.mknod(f'{folder}/input_current.dat')

    if not os.path.exists(f'{subfolder1}/raster.dat'):
        os.mknod(f'{subfolder1}/raster.dat')

    if not os.path.exists(f'{subfolder2}/raster.dat'):
        os.mknod(f'{subfolder2}/raster.dat')

    # Open data files
    v_file1 = open(f'{subfolder1}/v_avg.dat', 'w')
    v_file2 = open(f'{subfolder2}/v_avg.dat', 'w')
    r_file1 = open(f'{subfolder1}/r_avg.dat', 'w')
    r_file2 = open(f'{subfolder2}/r_avg.dat', 'w')
    s_file1 = open(f'{subfolder1}/s.dat', 'w')
    s_file2 = open(f'{subfolder2}/s.dat', 'w')
    z_file1 = open(f'{subfolder1}/z.dat', 'w')
    z_file2 = open(f'{subfolder2}/z.dat', 'w')
    input_file = open(f'{folder}/input_current.dat', 'w')
    raster_file1 = open(f'{subfolder1}/raster.dat', 'w')
    raster_file2 = open(f'{subfolder2}/raster.dat', 'w')

    for el in I :
        input_file.write(f'{el}, ')
    input_file.close()

# Read and processing the data
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

# Retrieve the data
def retrieve_files() :
    folder = '0'
    subfolder1 = '0/Population1'
    subfolder2 = '0/Population2'

    v1 = read_files(f'{subfolder1}/v_avg.dat')
    v2 = read_files(f'{subfolder2}/v_avg.dat')
    r1 = read_files(f'{subfolder1}/r_avg.dat')
    r2 = read_files(f'{subfolder2}/r_avg.dat')
    s1 = read_files(f'{subfolder1}/s.dat')
    s2 = read_files(f'{subfolder2}/s.dat')
    z1 = read_files(f'{subfolder1}/z.dat')
    z2 = read_files(f'{subfolder2}/z.dat')
    I = read_files(f'{folder}/input_current.dat')

    if os.path.isfile(f'{subfolder1}/raster.dat') :
        raster_file_1 = open(f'{subfolder1}/raster.dat', 'r')
        times_raster1, raster1 = [], []
        for line in raster_file_1 :
            times_raster1.append(float(line.split('  ')[0]))
            raster1.append(int(line.split('  ')[1].rstrip()))
    else :
        print(f'no file raster.dat in Population 1')
        exit()


    if os.path.isfile(f'{subfolder2}/raster.dat') :
        raster_file_2 = open(f'{subfolder2}/raster.dat', 'r')
        times_raster2, raster2 = [], []
        for line in raster_file_2 :
            times_raster2.append(float(line.split('  ')[0]))
            raster2.append(int(line.split('  ')[1].rstrip()))
    else :
        print(f'no file raster.dat in Population 2')
        exit()

    return v1, r1, s1, v2, r2, s2, I, times_raster1, raster1, times_raster2, raster2


''' NUMERICAL SIMULATION FUNCTIONS '''

''' POPULATION 1 '''
def QIF_pop1(t, v, r, s, v_avg, r_avg):
    dv = (v ** 2 + tau_m*Js[0]*s[0] + tau_m*Jc*s[1] + g[0]*(v_avg-v) + eta[0] + I[int(t/STEP)-1]) / tau_m
    ds = (-s[0] + r_avg) / tau_d
    return dv, ds

def make_step_pop1(t, v, r, s, v_avg, r_avg, spike_times):
    spike_times_next = np.copy(spike_times)

    # if spike_times == 0 & v >= vp
    spike_times_next[(spike_times == 0) & (v >= VP)] = t

    dv, ds = QIF_pop1(t, v, r, s, v_avg, r_avg)

    # if spike_times == 0 or v >= vp
    dv[(spike_times != 0 ) | (v > VP)] = 0
    v_next = v + dv*STEP
    s_next = s[0] + ds*STEP

    v_next[v >= VP] = VR

    spike_times_next[(spike_times != 0) & (t > spike_times + tau_s1)] = 0

    r_avg_next = np.mean((spike_times != 0) & (t <= spike_times + tau_s1))
    r_avg_next = r_avg_next/tau_s1
    v_avg_next = np.mean(v_next)

    spikes = spike_times_next[:selected_neurons] - spike_times[:selected_neurons]
    for k in range(len(spikes)) :
        if spikes[k] > 0 :
            raster_file1.write(f'{spike_times_next[k]}  {k}\n')

    return v_next, s_next, r_avg_next, v_avg_next, spike_times_next


''' POPULATION 2 '''
def QIF_pop2(t, v, r, s, v_avg, r_avg):
    dv = (v ** 2 + tau_m*Js[1]*s[1] + tau_m*Jc*s[0] + g[1]*(v_avg-v) + eta[1] + I[int(t/STEP)-1]) / tau_m
    ds = (-s[1] + r_avg) / tau_d
    return dv, ds

def make_step_pop2(t, v, r, s, v_avg, r_avg, spike_times):
    spike_times_next = np.copy(spike_times)

    # if spike_times == 0 & v >= vp
    spike_times_next[(spike_times == 0) & (v >= VP)] = t

    dv, ds = QIF_pop2(t, v, r, s, v_avg, r_avg)

    # if spike_times == 0 or v >= vp
    dv[(spike_times != 0 ) | (v > VP)] = 0
    v_next = v + dv*STEP
    s_next = s[1] + ds*STEP

    v_next[v >= VP] = VR

    spike_times_next[(spike_times != 0) & (t > spike_times + tau_s2)] = 0

    r_avg_next = np.mean((spike_times != 0) & (t <= spike_times + tau_s2))
    r_avg_next = r_avg_next/tau_s2
    v_avg_next = np.mean(v_next)

    spikes = spike_times_next[:selected_neurons] - spike_times[:selected_neurons]
    for k in range(len(spikes)) :
        if spikes[k] > 0 :
            raster_file2.write(f'{spike_times_next[k]}  {k}\n')

    return v_next, s_next, r_avg_next, v_avg_next, spike_times_next


''' RUN THE SIMULATIONS ON THE TWO POPULATIONS '''
def run():
    spike_times1, spike_times2 = np.zeros(N[0]), np.zeros(N[1])

    v1, v2 = v0[0], v0[1]
    r1, r2 =np.zeros(N[0]), np.zeros(N[1])
    s = s0

    v_avg1, v_avg2 = np.mean(v1), np.mean(v2)
    r_avg1, r_avg2 = np.mean(r1), np.mean(r2)

    for t in tqdm.tqdm(np.arange(0, T, STEP)):

        v_next1, s_next1, r_avg_next1, v_avg_next1, spike_times_next1 = make_step_pop1(t, v1, r1, s, v_avg1, r_avg1, spike_times1)
        v_next2, s_next2, r_avg_next2, v_avg_next2, spike_times_next2 = make_step_pop2(t, v2, r2, s, v_avg2, r_avg2, spike_times2)

        v1, v2 = v_next1, v_next2
        spike_times1, spike_times2 = spike_times_next1, spike_times_next2
        r_avg1, r_avg2 = r_avg_next1, r_avg_next2
        v_avg1, v_avg2 = v_avg_next1, v_avg_next2
        s = [s_next1, s_next2]

        w1 = math.pi*tau_m*r_avg1 + 1j*(v_avg1 - tau_m*math.log(a)*r_avg1)
        z1 = abs((1-w1.conjugate())/(1+w1.conjugate()))
        w2 = math.pi*tau_m*r_avg2 + 1j*(v_avg2 - tau_m*math.log(a)*r_avg2)
        z2 = abs((1-w2.conjugate())/(1+w2.conjugate()))

        # Save values on files
        v_file1.write(f'{np.round(v_avg1, 5)}, ')
        r_file1.write(f'{np.round(r_avg1, 5)}, ')
        s_file1.write(f'{np.round(s[0], 5)}, ')
        z_file1.write(f'{np.round(z1, 5)}, ')

        v_file2.write(f'{np.round(v_avg2, 5)}, ')
        r_file2.write(f'{np.round(r_avg2, 5)}, ')
        s_file2.write(f'{np.round(s[1], 5)}, ')
        z_file2.write(f'{np.round(z2, 5)}, ')


    v_file1.close()
    r_file1.close()
    s_file1.close()
    z_file1.close()
    raster_file1.close()

    v_file2.close()
    r_file2.close()
    s_file2.close()
    z_file2.close()
    raster_file2.close()


''' ANALYTICAL SOLUTION '''

def euler():
    v, r, s = [[np.mean(v0[0])], [np.mean(v0[1])]], [[r0[0]], [r0[1]]], [[s0[0]], [s0[1]]],
    print(v)
    print(r)
    print(s)
    a = 1
    for i in range(1, int(T/STEP)):
        r[0].append(r[0][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[0][i-1]*v[0][i-1] - g[0]*r[0][i-1] - 2*tau_m*math.log(a)*r[0][i-1]**2)/tau_m)
        r[1].append(r[1][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[1][i-1]*v[1][i-1] - g[1]*r[1][i-1] - 2*tau_m*math.log(a)*r[1][i-1]**2)/tau_m)
        v[0].append(v[0][i-1] + STEP*(v[0][i-1]**2 + eta_mean + Js[0]*tau_m*s[0][i-1] + Jc*tau_m*s[1][i-1]  + I[i-1] - (math.log(a)**2 + math.pi**2)*(tau_m*r[0][i-1])**2 + delta*math.log(a)/math.pi )/tau_m)
        v[1].append(v[1][i-1] + STEP*(v[1][i-1]**2 + eta_mean + Js[1]*tau_m*s[1][i-1] + Jc*tau_m*s[0][i-1]  + I[i-1] - (math.log(a)**2 + math.pi**2)*(tau_m*r[1][i-1])**2 + delta*math.log(a)/math.pi )/tau_m)
        s[0].append(s[0][i-1] + STEP*(-s[0][i-1] + r[0][i-1])/tau_d)
        s[1].append(s[1][i-1] + STEP*(-s[1][i-1] + r[1][i-1])/tau_d)
    return np.array(v[0]), np.array(v[1]), np.array(r[0]), np.array(r[1]), np.array(s[0]), np.array(s[1])

def R(z) :
    return (1/(math.pi*tau_m))*((1-z.conjugate())/(1+z.conjugate())).real

def V(z) :
    return ((1-z.conjugate())/(1+z.conjugate())).imag + tau_m*math.log(a)*R(z)

def dF1(z, s1, s2, I):
    dF1 = (1j/2) * ((z+1)**2) * (eta_mean + Js[0]*tau_m*s1 + Jc*tau_m*s2 + g[0]*V(z) + I) - ((z+1)**2)*delta/2 - ((1-z)**2)*(1j/2) + (1-z**2)*g[0]/2
    return dF1

def dF2(z, s1, s2, I):
    dF2 = (1j/2) * ((z+1)**2) * (eta_mean + Js[1]*tau_m*s1 + Jc*tau_m*s2 + g[1]*V(z) + I) - ((z+1)**2)*delta/2 - ((1-z)**2)*(1j/2) + (1-z**2)*g[1]/2
    return dF2

def euler_kuramoto():
    w1 = math.pi*tau_m*r0[0] + 1j *(np.mean(v0[0]) - tau_m*math.log(a)*r0[0])
    w2 = math.pi*tau_m*r0[1] + 1j *(np.mean(v0[1]) - tau_m*math.log(a)*r0[1])
    z1 = (1-w1.conjugate())/(1+w1.conjugate())
    z2 = (1-w2.conjugate())/(1+w2.conjugate())
    z = [[z1], [z2]]
    s = [[s0[0]], [s0[1]]]

    for i in range(1, int(T/STEP)) :
        df1 = dF1(z[0][i-1], s[0][i-1], s[1][i-1], I[i-1])
        df2 = dF2(z[1][i-1], s[1][i-1], s[0][i-1], I[i-1])
        z[0].append(z[0][i-1] + STEP*df1)
        z[1].append(z[1][i-1] + STEP*df2)
        ds1 = -s[0][i-1] + R(z[0][i-1])/tau_d
        ds2 = -s[1][i-1] + R(z[1][i-1])/tau_d
        s[0].append(s[0][i-1] + STEP*ds1)
        s[1].append(s[1][i-1] + STEP*ds2)

    z0, z1 = np.array(z[0]), np.array(z[1])
    return np.absolute(z0), np.absolute(z1)


''' PLOTTING '''

def plot_system(analytical=True) :
    # Generating subplots
    fig, ax = plt.subplots(5, 2, figsize=(10,7), sharex=True)

    # Retrieve simulation solutions
    v1, r1, s1, v2, r2, s2, I, times_raster1, raster1, times_raster2, raster2 = retrieve_files()
    selected_neurons = 1000

    # Simulation solutions
    # Plotting injected current I
    times = [float(STEP*k) for k in range(len(I))]
    ax[0,0].plot(times, I, color='black')
    ax[0,1].plot(times, I, color='black')
    ax[0,0].set_ylabel('$I(t)$')

    # Plotting voltage average
    times_v1 = [float(STEP*k) for k in range(len(v1))]
    times_v2 = [float(STEP*k) for k in range(len(v2))]
    ax[1,0].plot(times_v1, v1, c='k')
    ax[1,1].plot(times_v1, v2, c='k')
    #ax[1,0].set_ylim(-2.1, 1.9)
    #ax[1,1].set_ylim(-9, 11)
    ax[1,0].set_ylabel(r'$v_{avg}(t)$')

    # Plotting firing rate average
    times1 = [float(STEP*k) for k in range(len(r1))]
    times2 = [float(STEP*k) for k in range(len(r2))]
    ax[2,0].plot(times1, r1, c='k')
    ax[2,1].plot(times2, r2, c='k')
    #ax[2,1].set_yscale('log')
    #ax[2,1].set_ylim(0, 5.4)
    #ax[2,0].set_ylim(0, 1.1)
    ax[2,0].set_ylabel(r'$r(t)$')

    # Plotting synatic activation average
    times1 = [float(STEP*k) for k in range(len(s1))]
    times2 = [float(STEP*k) for k in range(len(s2))]
    ax[3,0].plot(times1, s1, c='k')
    ax[3,1].plot(times2, s2, c='k')
    #ax[3,1].set_ylim(0, 0.8)
    #ax[3,0].set_ylim(0, 0.5)
    ax[3,0].set_ylabel(r'$s(t)$')

    # Plotting raster plot
    ax[4,0].set_ylabel('neuron #')
    ax[4,0].scatter(times_raster1, raster1, s=0.5, alpha=0.5, c='k')
    ax[4,1].scatter(times_raster2, raster2, s=0.5, alpha=0.5, c='k')
    ax[4,0].set_ylim(0, selected_neurons)
    ax[4,1].set_ylim(0, selected_neurons)
    ax[4,0].set_yticks([0, 1000], [r'$0$', r'$1e3$'])
    ax[4,1].set_yticks([0, 1000], [r'$0$', r'$1e3$'])
    ax[4,0].set_xlabel('time')
    ax[4,1].set_xlabel('time')
    ax[4,0].set_xlim(0, T)
    ax[4,1].set_xlim(0, T)

    # Analytical solution
    if analytical :
        v_sol1, v_sol2, r_sol1, r_sol2, s_sol1, s_sol2 = euler()
        ax[1,0].plot([STEP*k for k in range(len(v_sol1))], v_sol1, c='r', label='analytical')
        ax[2,0].plot([STEP*k for k in range(len(r_sol1))], r_sol1, c='r')
        ax[3,0].plot([STEP*k for k in range(len(s_sol1))], s_sol1, c='r')

        ax[1,1].plot([STEP*k for k in range(len(v_sol2))], v_sol2, c='r', label='analytical')
        ax[2,1].plot([STEP*k for k in range(len(r_sol2))], r_sol2, c='r')
        ax[3,1].plot([STEP*k for k in range(len(s_sol2))], s_sol2, c='r')

    ax[0,0].set_title('Cluster 1', size=20)
    ax[0,1].set_title('Cluster 2', size=20)

    #plt.suptitle('Two networks of electrically and chemically coupled neurons')
    plt.tight_layout()
    fig.align_ylabels(ax)
    plt.subplots_adjust(hspace=0.5)

    #plt.savefig(f'{folder}/full_network.png', dpi=600)

    plt.show()
    plt.close()


plot_system(False)