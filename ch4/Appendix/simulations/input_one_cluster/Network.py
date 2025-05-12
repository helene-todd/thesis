#!/usr/bin/env python
# coding: utf-8

from jax import jit, numpy as jnp
import numpy as np
import math, jax, tqdm, os, matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

np.random.seed(42)

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


class QIFNetwork() :

    def __init__(self, T, N, tau_m, tau_d, tau_s1, tau_s2, I, Js, Jc, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, s0, r0):
        self.selected_neurons = 1000
        self.I = I
        self.T = T
        self.N = N
        self.tau_m = tau_m
        self.tau_d = tau_d
        self.tau_s1 = tau_s1
        self.tau_s2 = tau_s2
        self.g = g #(2)
        self.Js = Js #(2)
        self.Jc = Jc #(1)
        self.VP = VP
        self.VR = VR
        self.a = abs(VP/VR)
        self.REFRACT_TIME = REFRACT_TIME
        self.STEP = STEP
        self.delta = delta
        self.eta = [delta*np.tan(math.pi*(np.random.random(N[0])-0.5))+eta_mean, delta*np.tan(math.pi*(np.random.random(N[1])-0.5))+eta_mean]
        self.eta_mean = eta_mean
        self.v0, self.s0, self.r0 = v0, s0, r0
        self.create_directories()
        print('network initialized.')


    ''' GENERATE DIRECTORY & FILES FUNCTIONS '''

    # Create subdirectory for future files
    def create_directories(self) :
        directory_list = list()
        for root, dirs, files in os.walk(os.getcwd(), topdown=False):
            for name in dirs:
                directory_list.append(os.path.join(name))
        folder_nb = 0
        for d in directory_list :
            if d.isdigit() and int(d) >= folder_nb:
                folder_nb = int(d)+1

        # Create data files in new subdirectory
        self.folder = str(folder_nb)
        self.subfolder1 = str(folder_nb)+'/Population1'
        self.subfolder2 = str(folder_nb)+'/Population2'

        if not os.path.exists(os.path.join(os.getcwd(), self.subfolder1)):
            os.makedirs(os.path.join(os.getcwd(), self.subfolder1))

        if not os.path.exists(os.path.join(os.getcwd(), self.subfolder2)):
            os.makedirs(os.path.join(os.getcwd(), self.subfolder2))

        # Open data files
        self.v_file1 = open(f'{self.subfolder1}/v_avg.dat', 'w')
        self.v_file2 = open(f'{self.subfolder2}/v_avg.dat', 'w')
        self.r_file1 = open(f'{self.subfolder1}/r_avg.dat', 'w')
        self.r_file2 = open(f'{self.subfolder2}/r_avg.dat', 'w')
        self.s_file1 = open(f'{self.subfolder1}/s.dat', 'w')
        self.s_file2 = open(f'{self.subfolder2}/s.dat', 'w')
        self.input_file1 = open(f'{self.subfolder1}/input_current.dat', 'w')
        self.input_file2 = open(f'{self.subfolder2}/input_current.dat', 'w')
        self.raster_file1 = open(f'{self.subfolder1}/raster.dat', 'w')
        self.raster_file2 = open(f'{self.subfolder2}/raster.dat', 'w')

        for el in self.I[0] :
            self.input_file1.write(f'{el}, ')
        self.input_file1.close()

        for el in self.I[1] :
            self.input_file2.write(f'{el}, ')
        self.input_file2.close()

    # Read and processing the data
    def read_files(self, filename):
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
    def retrieve_files(self) :
        self.v1 = self.read_files(f'{self.subfolder1}/v_avg.dat')
        self.v2 = self.read_files(f'{self.subfolder2}/v_avg.dat')
        self.r1 = self.read_files(f'{self.subfolder1}/r_avg.dat')
        self.r2 = self.read_files(f'{self.subfolder2}/r_avg.dat')
        self.s1 = self.read_files(f'{self.subfolder1}/s.dat')
        self.s2 = self.read_files(f'{self.subfolder2}/s.dat')
        self.input1 = self.read_files(f'{self.subfolder1}/input_current.dat')
        self.input2 = self.read_files(f'{self.subfolder2}/input_current.dat')

        if os.path.isfile(f'{self.subfolder1}/raster.dat') :
            self.raster_file_1 = open(f'{self.subfolder1}/raster.dat', 'r')
            self.times_raster1, self.raster1 = [], []
            for line in self.raster_file_1 :
                self.times_raster1.append(float(line.split('  ')[0]))
                self.raster1.append(int(line.split('  ')[1].rstrip()))
        else :
            print(f'no file raster.dat in Population 1')
            exit()

        if os.path.isfile(f'{self.subfolder2}/raster.dat') :
            self.raster_file_2 = open(f'{self.subfolder2}/raster.dat', 'r')
            self.times_raster2, self.raster2 = [], []
            for line in self.raster_file_2 :
                self.times_raster2.append(float(line.split('  ')[0]))
                self.raster2.append(int(line.split('  ')[1].rstrip()))
        else :
            print(f'no file raster.dat in Population 2')
            exit()


    ''' NUMERICAL SIMULATION FUNCTIONS '''

    ''' POPULATION 1 '''
    def QIF_pop1(self, t, v, r, s, v_avg, r_avg):
        dv = (v ** 2 + self.tau_m*self.Js[0]*s[0] + self.tau_m*self.Jc*s[1] + self.g[0]*(v_avg-v) + self.eta[0] + 0*self.I[0][int(t/self.STEP)-1]) / self.tau_m
        ds = (-s[0] + r_avg) / self.tau_d
        return dv, ds

    def make_step_pop1(self, t, v, r, s, v_avg, r_avg, spike_times):
        spike_times_next = np.copy(spike_times)

        # if spike_times == 0 & v >= vp
        spike_times_next[(spike_times == 0) & (v >= self.VP)] = t
        dv, ds = self.QIF_pop1(t, v, r, s, v_avg, r_avg)

        # if spike_times == 0 or v >= vp
        dv[(spike_times != 0 ) | (v > self.VP)] = 0

        v_next = v + dv*self.STEP
        s_next = s[0] + ds*self.STEP

        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME)] = self.VP
        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t > spike_times + self.REFRACT_TIME)] = self.VR
        v_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = self.VR
        spike_times_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = 0
        r_avg_next = np.mean((spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t < spike_times + self.tau_s1))/self.tau_s1
        v_avg_next = np.mean(v_next)

        spikes = spike_times_next[:self.selected_neurons] - spike_times[:self.selected_neurons]
        for k in range(len(spikes)) :
            if spikes[k] > 0 :
                self.raster_file1.write(f'{spike_times_next[k]}  {k}\n')

        return v_next, s_next, r_avg_next, v_avg_next, spike_times_next


    ''' POPULATION 2 '''
    def QIF_pop2(self, t, v, r, s, v_avg, r_avg):
        dv = (v ** 2 + self.tau_m*self.Js[1]*s[1] + self.tau_m*self.Jc*s[0] + self.g[1]*(v_avg-v) + self.eta[1] + self.I[1][int(t/self.STEP)-1]) / self.tau_m
        ds = (-s[1] + r_avg) / self.tau_d
        return dv, ds

    def make_step_pop2(self, t, v, r, s, v_avg, r_avg, spike_times):
        spike_times_next = np.copy(spike_times)

        # if spike_times == 0 & v >= vp
        spike_times_next[(spike_times == 0) & (v >= self.VP)] = t
        dv, ds = self.QIF_pop2(t, v, r, s, v_avg, r_avg)

        # if spike_times == 0 or v >= vp
        dv[(spike_times != 0 ) | (v > self.VP)] = 0

        v_next = v + dv*self.STEP
        s_next = s[1] + ds*self.STEP
        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME)] = self.VP
        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t > spike_times + self.REFRACT_TIME)] = self.VR
        v_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = self.VR
        spike_times_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = 0
        r_avg_next = np.mean((spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t < spike_times + self.tau_s2))/self.tau_s2
        v_avg_next = np.mean(v_next)

        spikes = spike_times_next[:self.selected_neurons] - spike_times[:self.selected_neurons]
        for k in range(len(spikes)) :
            if spikes[k] > 0 :
                self.raster_file2.write(f'{spike_times_next[k]}  {k}\n')

        return v_next, s_next, r_avg_next, v_avg_next, spike_times_next


    ''' RUN THE SIMULATIONS ON THE TWO POPULATIONS '''
    def run(self):
        spike_times1, spike_times2 = np.zeros(self.N[0]), np.zeros(self.N[1])

        v1, v2 = self.v0[0], self.v0[1]
        r1, r2 =np.zeros(self.N[0]), np.zeros(self.N[1])
        s = self.s0

        v_avg1, v_avg2 = np.mean(v1), np.mean(v2)
        r_avg1, r_avg2 = np.mean(r1), np.mean(r2)

        for t in tqdm.tqdm(np.arange(0, self.T, self.STEP)):

            v_next1, s_next1, r_avg_next1, v_avg_next1, spike_times_next1 = self.make_step_pop1(t, v1, r1, s, v_avg1, r_avg1, spike_times1)
            v_next2, s_next2, r_avg_next2, v_avg_next2, spike_times_next2 = self.make_step_pop2(t, v2, r2, s, v_avg2, r_avg2, spike_times2)

            v1, v2 = v_next1, v_next2
            spike_times1, spike_times2 = spike_times_next1, spike_times_next2
            r_avg1, r_avg2 = r_avg_next1, r_avg_next2
            v_avg1, v_avg2 = v_avg_next1, v_avg_next2
            s = [s_next1, s_next2]

            # Save values on files
            self.v_file1.write(f'{np.round(v_avg1, 5)}, ')
            self.r_file1.write(f'{np.round(r_avg1, 5)}, ')
            self.s_file1.write(f'{np.round(s[0], 5)}, ')

            self.v_file2.write(f'{np.round(v_avg2, 5)}, ')
            self.r_file2.write(f'{np.round(r_avg2, 5)}, ')
            self.s_file2.write(f'{np.round(s[1], 5)}, ')


        self.v_file1.close()
        self.r_file1.close()
        self.s_file1.close()
        self.raster_file1.close()

        self.v_file2.close()
        self.r_file2.close()
        self.s_file2.close()
        self.raster_file2.close()


    ''' ANALYTICAL SOLUTION '''
    def euler(self):
        v, r, s = [[np.mean(self.v0[0])], [np.mean(self.v0[1])]], [[self.r0[0]], [self.r0[1]]], [[self.s0[0]], [self.s0[1]]],
        for i in range(1, int(self.T/self.STEP)):
            r[0].append(r[0][i-1] + self.STEP*(self.delta/(self.tau_m*math.pi) + 2*r[0][i-1]*v[0][i-1] - self.g[0]*r[0][i-1] - 2*self.tau_m*math.log(self.a)*r[0][i-1]**2)/self.tau_m)
            r[1].append(r[1][i-1] + self.STEP*(self.delta/(self.tau_m*math.pi) + 2*r[1][i-1]*v[1][i-1] - self.g[1]*r[1][i-1] - 2*self.tau_m*math.log(self.a)*r[1][i-1]**2)/self.tau_m)
            v[0].append(v[0][i-1] + self.STEP*(v[0][i-1]**2 + self.eta_mean + self.Js[0]*self.tau_m*s[0][i-1] + self.Jc*self.tau_m*s[1][i-1]  + self.I[0][i-1] - (math.log(self.a)**2 + math.pi**2)*(self.tau_m*r[0][i-1])**2 + self.delta*math.log(self.a)/math.pi )/self.tau_m)
            v[1].append(v[1][i-1] + self.STEP*(v[1][i-1]**2 + self.eta_mean + self.Js[1]*self.tau_m*s[1][i-1] + self.Jc*self.tau_m*s[0][i-1]  + self.I[1][i-1] - (math.log(self.a)**2 + math.pi**2)*(self.tau_m*r[1][i-1])**2 + self.delta*math.log(self.a)/math.pi )/self.tau_m)
            s[0].append(s[0][i-1] + self.STEP*(-s[0][i-1] + r[0][i-1])/self.tau_d)
            s[1].append(s[1][i-1] + self.STEP*(-s[1][i-1] + r[1][i-1])/self.tau_d)
        return np.array(v[0]), np.array(v[1]), np.array(r[0]), np.array(r[1]), np.array(s[0]), np.array(s[1])

    ''' PLOTTING '''
    def plot_system(self, analytical=True) :
        # Generating subplots
        fig, ax = plt.subplots(5, 2, figsize=(12,6), sharex=True)

        # Retrieve simulation solutions
        self.retrieve_files()

        # Simulation solutions
        # Plotting injected current I
        time = np.linspace(0, self.T, self.STEP)
        ax[0,0].plot(time, self.input1[:time], linewidth=2, color='black')
        ax[0,1].plot(time, self.input2[:time], linewidth=2, color='black')
        ax[0,0].set_ylabel('$I(t)$')

        # Plotting voltage average
        #times_v1 = [float(self.STEP*k) for k in range(len(self.v1))]
        #times_v2 = [float(self.STEP*k) for k in range(len(self.v2))]
        ax[1,0].plot(times_v1, self.v1[:time], linewidth=2, c='k')
        ax[1,1].plot(times_v1, self.v2[:time], linewidth=2, c='k')
        ax[1,0].set_ylabel(r'$v_{avg}(t)$')

        # Plotting firing rate average
        times1 = [float(self.STEP*k) for k in range(len(self.r1))]
        times2 = [float(self.STEP*k) for k in range(len(self.r2))]
        ax[2,0].plot(times1, self.r1, linewidth=2, c='k')
        ax[2,1].plot(times2, self.r2, linewidth=2, c='k')
        #ax[2,1].set_yscale('log')
        #ax[2,0].set_yscale('log')
        ax[2,1].set_ylim(0, 20)
        ax[2,0].set_ylim(0, 4)
        ax[2,0].set_ylabel(r'$r(t)$')

        # Plotting synatic activation average
        times1 = [float(self.STEP*k) for k in range(len(self.s1))]
        times2 = [float(self.STEP*k) for k in range(len(self.s2))]
        ax[3,0].plot(times1, self.s1, linewidth=2, c='k')
        ax[3,1].plot(times2, self.s2, linewidth=2, c='k')
        #ax[3,1].set_ylim(0, 0.9)
        #ax[3,0].set_ylim(0, 0.3)
        ax[3,0].set_ylabel(r'$s(t)$')

        # Plotting raster plot
        ax[4,0].set_ylabel('neuron #')
        ax[4,0].scatter(self.times_raster1, self.raster1, s=0.75, alpha=0.5, c='k')
        ax[4,1].scatter(self.times_raster2, self.raster2, s=0.75, alpha=0.5, c='k')
        ax[4,0].set_ylim(0, self.selected_neurons)
        ax[4,1].set_ylim(0, self.selected_neurons)
        ax[4,0].set_yticks([0, 1000], [r'$0$', r'$10e3$'])
        ax[4,1].set_yticks([0, 1000], [r'$0$', r'$10e3$'])
        ax[4,0].set_xlabel('time')
        ax[4,1].set_xlabel('time')
        ax[4,0].set_xlim(0, self.T)
        ax[4,1].set_xlim(0, self.T)

        # Analytical solution
        if analytical :
            v_sol1, v_sol2, r_sol1, r_sol2, s_sol1, s_sol2 = self.euler()
            ax[1,0].plot([self.STEP*k for k in range(len(v_sol1))], v_sol1, linewidth=2, c='r', label='analytical')
            ax[2,0].plot([self.STEP*k for k in range(len(r_sol1))], r_sol1, linewidth=2, c='r')
            ax[3,0].plot([self.STEP*k for k in range(len(s_sol1))], s_sol1, linewidth=2, c='r')

            ax[1,1].plot([self.STEP*k for k in range(len(v_sol2))], v_sol2, linewidth=2, c='r', label='analytical')
            ax[2,1].plot([self.STEP*k for k in range(len(r_sol2))], r_sol2, linewidth=2, c='r')
            ax[3,1].plot([self.STEP*k for k in range(len(s_sol2))], s_sol2, linewidth=2, c='r')

        ax[0,0].set_title('Cluster 1', size=20)
        ax[0,1].set_title('Cluster 2', size=20)

        plt.tight_layout()
        fig.align_ylabels(ax)
        plt.subplots_adjust(hspace=0.5)

        plt.savefig(f'{self.folder}/full_network.png', dpi=600)

        plt.show()
        plt.close()
