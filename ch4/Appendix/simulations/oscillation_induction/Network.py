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
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

green = '#119a7bfe'
purple = '#813d9cfe'

class QIFNetwork() :

    def __init__(self, T, N, tau_m, tau_d, tau_s1, tau_s2, I, Js, Jc, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, s0, r0):
        self.selected_neurons = 1000
        self.I = I

        self.create_directories()

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

        if not os.path.exists(f'{self.subfolder1}/v_avg.dat'):
            os.mknod(f'{self.subfolder1}/v_avg_1.dat')

        if not os.path.exists(f'{self.subfolder2}/v_avg.dat'):
            os.mknod(f'{self.subfolder2}/v_avg_2.dat')

        if not os.path.exists(f'{self.subfolder1}/r_avg.dat'):
            os.mknod(f'{self.subfolder1}/r_avg.dat')

        if not os.path.exists(f'{self.subfolder2}/r_avg.dat'):
            os.mknod(f'{self.subfolder2}/r_avg.dat')

        if not os.path.exists(f'{self.subfolder1}/s.dat'):
            os.mknod(f'{self.subfolder1}/s.dat')

        if not os.path.exists(f'{self.subfolder2}/s.dat'):
            os.mknod(f'{self.subfolder2}/s.dat')

        if not os.path.exists(f'{self.subfolder1}/z.dat'):
            os.mknod(f'{self.subfolder1}/z.dat')

        if not os.path.exists(f'{self.subfolder2}/z.dat'):
            os.mknod(f'{self.subfolder2}/z.dat')

        if not os.path.exists(f'{self.folder}/input_current.dat'):
            os.mknod(f'{self.folder}/input_current.dat')

        if not os.path.exists(f'{self.subfolder1}/raster.dat'):
            os.mknod(f'{self.subfolder1}/raster.dat')

        if not os.path.exists(f'{self.subfolder2}/raster.dat'):
            os.mknod(f'{self.subfolder2}/raster.dat')

        # Open data files
        self.v_file1 = open(f'{self.subfolder1}/v_avg.dat', 'w')
        self.v_file2 = open(f'{self.subfolder2}/v_avg.dat', 'w')
        self.r_file1 = open(f'{self.subfolder1}/r_avg.dat', 'w')
        self.r_file2 = open(f'{self.subfolder2}/r_avg.dat', 'w')
        self.s_file1 = open(f'{self.subfolder1}/s.dat', 'w')
        self.s_file2 = open(f'{self.subfolder2}/s.dat', 'w')
        self.z_file1 = open(f'{self.subfolder1}/z.dat', 'w')
        self.z_file2 = open(f'{self.subfolder2}/z.dat', 'w')
        self.input_file = open(f'{self.folder}/input_current.dat', 'w')
        self.raster_file1 = open(f'{self.subfolder1}/raster.dat', 'w')
        self.raster_file2 = open(f'{self.subfolder2}/raster.dat', 'w')

        for el in self.I :
            self.input_file.write(f'{el}, ')
        self.input_file.close()

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
        self.z1 = self.read_files(f'{self.subfolder1}/z.dat')
        self.z2 = self.read_files(f'{self.subfolder2}/z.dat')
        self.input = self.read_files(f'{self.folder}/input_current.dat')

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
        dv = (v ** 2 + self.tau_m*self.Js[0]*s[0] + self.tau_m*self.Jc*s[1] + self.g[0]*(v_avg-v) + self.eta[0] + self.I[int(t/self.STEP)-1]) / self.tau_m
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
        dv = (v ** 2 + self.tau_m*self.Js[1]*s[1] + self.tau_m*self.Jc*s[0] + self.g[1]*(v_avg-v) + self.eta[1] + self.I[int(t/self.STEP)-1]) / self.tau_m
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

            w1 = math.pi*self.tau_m*r_avg1 + 1j*(v_avg1 - self.tau_m*math.log(self.a)*r_avg1)
            z1 = abs((1-w1.conjugate())/(1+w1.conjugate()))
            w2 = math.pi*self.tau_m*r_avg2 + 1j*(v_avg2 - self.tau_m*math.log(self.a)*r_avg2)
            z2 = abs((1-w2.conjugate())/(1+w2.conjugate()))

            # Save values on files
            self.v_file1.write(f'{np.round(v_avg1, 5)}, ')
            self.r_file1.write(f'{np.round(r_avg1, 5)}, ')
            self.s_file1.write(f'{np.round(s[0], 5)}, ')
            self.z_file1.write(f'{np.round(z1, 5)}, ')

            self.v_file2.write(f'{np.round(v_avg2, 5)}, ')
            self.r_file2.write(f'{np.round(r_avg2, 5)}, ')
            self.s_file2.write(f'{np.round(s[1], 5)}, ')
            self.z_file2.write(f'{np.round(z2, 5)}, ')

            if t == self.T/2 :
                self.Jc = -1


        self.v_file1.close()
        self.r_file1.close()
        self.s_file1.close()
        self.z_file1.close()
        self.raster_file1.close()

        self.v_file2.close()
        self.r_file2.close()
        self.s_file2.close()
        self.z_file2.close()
        self.raster_file2.close()

    ''' PLOTTING '''

    def plot_system(self) :
        # Generating subplots
        fig, ax = plt.subplots(4, 1, figsize=(14,8), sharex=True)

        # Retrieve simulation solutions
        self.retrieve_files()

        lw=1.8

        # Plotting voltage average
        times_v1 = [float(self.STEP*k) for k in range(len(self.v1))]
        times_v2 = [float(self.STEP*k) for k in range(len(self.v2))]

        ax[0].plot(times_v1, self.v1, c=purple, linewidth=2.2, label='cluster 1')
        ax[0].plot(times_v2, self.v2, c=green, linewidth=2.2, label='cluster 2')
        ax[0].set_ylim(-3, 4)
        ax[0].set_ylabel(r'$v_{avg}(t)$')

        # Plotting firing rate average
        times1 = [float(self.STEP*k) for k in range(len(self.r1))]
        times2 = [float(self.STEP*k) for k in range(len(self.r2))]
        ax[1].plot(times1, self.r1, c=purple, linewidth=lw)
        ax[1].plot(times2, self.r2, c=green, linewidth=lw)
        ax[1].set_ylabel(r'$r(t)$')

        # Plotting synatic activation average
        times1 = [float(self.STEP*k) for k in range(len(self.s1))]
        times2 = [float(self.STEP*k) for k in range(len(self.s2))]
        ax[2].plot(times1, self.s1, c=purple, linewidth=lw)
        ax[2].plot(times2, self.s2, c=green, linewidth=lw)
        ax[2].set_ylabel(r'$s(t)$')

        # Plotting raster plot
        ax[3].set_ylabel('neuron #')
        ax[3].scatter(self.times_raster1, self.raster1, s=1, alpha=0.5, c=purple)
        ax[3].scatter(self.times_raster2, self.raster2, s=1, alpha=0.5, c=green)
        ax[3].set_ylim(0, self.selected_neurons)
        ax[3].set_yticks([0, 500, 1000], ['0', '', '1e3'])
        ax[3].set_xlabel('time')
        ax[3].set_xlim(0, self.T)

        ax[0].legend(loc='upper left')

        fig.align_ylabels(ax)

        #plt.suptitle('Two networks of electrically and chemically coupled neurons')
        plt.tight_layout()

        plt.savefig(f'{self.folder}/full_network.png', dpi=600)

        plt.show()
        plt.close()
