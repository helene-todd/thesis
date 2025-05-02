#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math, jax, tqdm, os, matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#sd=16, 31, 33
sd=6
np.random.seed(sd)

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


class QIFNetwork() :

    def __init__(self, T, N, tau_m, tau_s, J, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, r0):
        self.selected_neurons = 1000

        self.create_directories()

        self.T = T
        self.N = N

        self.tau_m = tau_m
        self.tau_s = tau_s

        self.g = g
        self.J = J

        self.VP = VP
        self.VR = VR
        self.REFRACT_TIME = REFRACT_TIME
        self.STEP = STEP

        self.delta = delta
        self.eta = np.sort(delta*np.tan(math.pi*(np.random.random(N)-0.5))+eta_mean)
        np.random.shuffle(self.eta)
        self.eta_mean = eta_mean

        self.v0, self.r0 = v0, r0

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
        self.subfolder = str(folder_nb)
        if not os.path.exists(os.path.join(os.getcwd(), self.subfolder)):
            os.makedirs(os.path.join(os.getcwd(), self.subfolder))

        if not os.path.exists(f'{self.subfolder}/v_avg.dat'):
            os.mknod(f'{self.subfolder}/v_avg.dat')

        if not os.path.exists(f'{self.subfolder}/r_avg.dat'):
            os.mknod(f'{self.subfolder}/r_avg.dat')

        if not os.path.exists(f'{self.subfolder}/raster.dat'):
            os.mknod(f'{self.subfolder}/raster.dat')

        # Open data files
        self.v_file = open(f'{self.subfolder}/v_avg.dat', 'w')
        self.r_file = open(f'{self.subfolder}/r_avg.dat', 'w')
        self.raster_file = open(f'{self.subfolder}/raster.dat', 'w')

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
        self.v = self.read_files(f'{self.subfolder}/v_avg.dat')
        self.r = self.read_files(f'{self.subfolder}/r_avg.dat')

        if os.path.isfile(f'{self.subfolder}/raster.dat') :
            self.raster_file = open(f'{self.subfolder}/raster.dat', 'r')
            self.times_raster, self.raster = [], []
            for line in self.raster_file :
                self.times_raster.append(float(line.split('  ')[0]))
                self.raster.append(int(line.split('  ')[1].rstrip()))
        else :
            print(f'no file raster.dat')
            exit()


    ''' NUMERICAL SIMULATION FUNCTIONS '''

    # ODE system
    def QIF(self, t, v, v_avg, r_avg):
        dv = (v ** 2 + self.tau_m*self.J*r_avg + self.g*(v_avg-v) + self.eta) / self.tau_m
        return dv


    # One time step
    def make_step(self, t, v, v_avg, r_avg, spike_times):
        spike_times_next = np.copy(spike_times)

        # if spike_times == 0 & v >= vp
        spike_times_next[(spike_times == 0) & (v >= self.VP)] = t

        dv = self.QIF(t, v, v_avg, r_avg)

        dv[(spike_times != 0 ) | (v > self.VP)] = 0
        v_next = v + dv*self.STEP

        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME)] = self.VP

        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t > spike_times + self.REFRACT_TIME)] = self.VR

        v_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = self.VR
        spike_times_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = 0

        r_avg_next = np.sum((spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t < spike_times + self.tau_s))/(self.N*self.tau_s)
        v_avg_next = np.mean(v_next)

        spikes = spike_times_next[:self.selected_neurons] - spike_times[:self.selected_neurons]
        for k in range(len(spikes)) :
            if spikes[k] > 0 :
                self.raster_file.write(f'{spike_times_next[k]}  {k}\n')

        return v_next, r_avg_next, v_avg_next, spike_times_next


    # Run all time steps of simulation
    def run(self):
        spike_times = np.zeros(self.N)

        v = self.v0

        v_avg = np.mean(v)
        r_avg = 0

        for t in tqdm.tqdm(np.arange(0, self.T, self.STEP)):

            v_next, r_avg_next, v_avg_next, spike_times_next = self.make_step(t, v, v_avg, r_avg, spike_times)

            v = v_next
            r_avg = r_avg_next
            v_avg = v_avg_next
            spike_times = spike_times_next

            # Save values in files
            self.v_file.write(f'{np.round(v_avg, 5)}, ')
            self.r_file.write(f'{np.round(r_avg, 5)}, ')

        self.v_file.close()
        self.r_file.close()
        self.raster_file.close()

    ''' ANALYTICAL SOLUTION '''

    def euler(self):
        v, r = [np.mean(self.v0)], [self.r0]
        for i in range(1, int(self.T/self.STEP)):
            r.append(r[i-1] + self.STEP*(self.delta/(self.tau_m*math.pi) + 2*r[i-1]*v[i-1] - self.g*r[i-1] )/self.tau_m)
            v.append(v[i-1] + self.STEP*(v[i-1]**2 + self.eta_mean + self.J*self.tau_m*r[i-1] -  (math.pi**2)*(self.tau_m*r[i-1])**2)/self.tau_m)
        return np.array(v), np.array(r)

    ''' PLOTTING '''

    def plot_system(self, analytical=True) :
        # Generating subplots
        fig, ax = plt.subplots(3, 1, figsize=(12,7), sharex=True)

        # Retrieve simulation solutions
        self.retrieve_files()

        # Plotting voltage average
        times = [float(self.STEP*k) for k in range(len(self.v))]
        #ax[0].plot(times, self.v, c='k')
        ax[0].set_ylabel(r'$v(t)$')

        # Plotting firing rate average
        times = [float(self.STEP*k) for k in range(len(self.r))]
        #ax[1].plot(times, self.r, c='k')
        ax[1].set_ylabel(r'$r(t)$')
        #ax[1].set_ylim(0, 0.3)

        # Plotting raster plot
        ax[2].set_ylabel('neuron #')
        ax[2].scatter(self.times_raster, self.raster, s=0.9, alpha=0.25, c='k')
        ax[2].set_ylim(0, self.selected_neurons)
        ax[2].set_yticks([0, 1000], [r'$0$', r'$1e3$'])
        ax[2].set_xlabel('time')
        ax[2].set_xlim(0, self.T)

        # Analytical solution
        if analytical :
            v_sol, r_sol = self.euler()
            ax[0].plot([self.STEP*k for k in range(len(v_sol))], v_sol, c='r')
            ax[1].plot([self.STEP*k for k in range(len(r_sol))], r_sol, c='r')

        #plt.suptitle('Network of 10e4 electrically and chemically coupled neurons')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)

        plt.savefig(f'{self.subfolder}/full_network.png', dpi=600)

        plt.show()
        plt.close()
