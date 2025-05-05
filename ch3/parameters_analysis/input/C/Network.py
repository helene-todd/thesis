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


class QIFNetwork() :

    def __init__(self, T, N, tau_m, tau_d, tau_s, I, J, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, s0, r0):
        self.selected_neurons = N
        self.I = I

        self.create_directories()

        self.T = T
        self.N = N

        self.tau_m = tau_m
        self.tau_d = tau_d
        self.tau_s = tau_s

        self.g = g
        self.J = J

        self.VP = VP
        self.VR = VR
        self.a = abs(VP/VR)
        self.REFRACT_TIME = REFRACT_TIME
        self.STEP = STEP

        self.delta = delta
        self.eta = np.sort(delta*np.tan(math.pi*(np.random.random(N)-0.5))+eta_mean)
        np.random.shuffle(self.eta)
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
        self.subfolder = str(folder_nb)
        if not os.path.exists(os.path.join(os.getcwd(), self.subfolder)):
            os.makedirs(os.path.join(os.getcwd(), self.subfolder))

        if not os.path.exists(f'{self.subfolder}/v_avg.dat'):
            os.mknod(f'{self.subfolder}/v_avg.dat')

        if not os.path.exists(f'{self.subfolder}/r_avg.dat'):
            os.mknod(f'{self.subfolder}/r_avg.dat')

        if not os.path.exists(f'{self.subfolder}/z_avg.dat'):
            os.mknod(f'{self.subfolder}/z_avg.dat')

        if not os.path.exists(f'{self.subfolder}/input_current.dat'):
            os.mknod(f'{self.subfolder}/input_current.dat')

        if not os.path.exists(f'{self.subfolder}/raster.dat'):
            os.mknod(f'{self.subfolder}/raster.dat')

        # Open data files
        self.v_file = open(f'{self.subfolder}/v_avg.dat', 'w')
        self.r_file = open(f'{self.subfolder}/r_avg.dat', 'w')
        self.s_file = open(f'{self.subfolder}/s.dat', 'w')
        self.z_file = open(f'{self.subfolder}/z.dat', 'w')
        self.input_file = open(f'{self.subfolder}/input_current.dat', 'w')
        self.raster_file = open(f'{self.subfolder}/raster.dat', 'w')

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
        self.v = self.read_files(f'{self.subfolder}/v_avg.dat')
        self.r = self.read_files(f'{self.subfolder}/r_avg.dat')
        self.s = self.read_files(f'{self.subfolder}/s.dat')
        self.z = self.read_files(f'{self.subfolder}/z.dat')
        self.input = self.read_files(f'{self.subfolder}/input_current.dat')

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
    def QIF(self, t, v, r, s, v_avg, r_avg):
        dv = (v ** 2 + self.tau_m*self.J*s + self.g*(v_avg-v) + self.eta + self.I[int(t/self.STEP)-1]) / self.tau_m
        ds = (-s + r_avg) / self.tau_d
        return dv, ds


    # One time step
    def make_step(self, t, v, r, s, v_avg, r_avg, spike_times):
        spike_times_next = np.copy(spike_times)

        # if spike_times == 0 & v >= vp
        spike_times_next[(spike_times == 0) & (v >= self.VP)] = t

        dv, ds = self.QIF(t, v, r, s, v_avg, r_avg)

        # if spike_times == 0 or v >= vp
        dv[(spike_times != 0 ) | (v > self.VP)] = 0
        v_next = v + dv*self.STEP
        s_next = s + ds*self.STEP

        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME)] = self.VP

        v_next[(spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t > spike_times + self.REFRACT_TIME)] = self.VR

        v_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = self.VR
        spike_times_next[(spike_times != 0) & (t > spike_times + 2 * self.REFRACT_TIME)] = 0

        r_avg_next = np.mean((spike_times != 0) & (t <= spike_times + 2 * self.REFRACT_TIME) & (t < spike_times + self.tau_s))/self.tau_s
        v_avg_next = np.mean(v_next)

        spikes = spike_times_next[:self.selected_neurons] - spike_times[:self.selected_neurons]
        for k in range(len(spikes)) :
            if spikes[k] > 0 :
                self.raster_file.write(f'{spike_times_next[k]}  {k}\n')

        return v_next, s_next, r_avg_next, v_avg_next, spike_times_next


    # Run all time steps of simulation
    def run(self):
        spike_times = np.zeros(self.N)

        v = self.v0
        r = np.zeros(self.N)
        s = self.s0

        v_avg = np.mean(v)
        r_avg = np.mean(r)

        for t in tqdm.tqdm(np.arange(0, self.T, self.STEP)):

            v_next, s_next, r_avg_next, v_avg_next, spike_times_next = self.make_step(t, v, r, s, v_avg, r_avg, spike_times)

            v = v_next
            s = s_next
            r_avg = r_avg_next
            v_avg = v_avg_next
            spike_times = spike_times_next

            w = math.pi*self.tau_m*r_avg + 1j*(v_avg - self.tau_m*math.log(self.a)*r_avg)
            z = abs((1-w.conjugate())/(1+w.conjugate()))

            # Save values on files
            self.v_file.write(f'{np.round(v_avg, 5)}, ')
            self.r_file.write(f'{np.round(r_avg, 5)}, ')
            self.s_file.write(f'{np.round(s, 5)}, ')
            self.z_file.write(f'{np.round(z, 5)}, ')

            #spikes = (v[1:-1] < self.VP) & (v[2:] >= self.VP)
            #events = [np.where(s)[0] for s in spikes.T]
            #for el in events :
                #self.raster_file.write(f'{t}  {el}\n')

        self.v_file.close()
        self.r_file.close()
        self.s_file.close()
        self.z_file.close()
        self.raster_file.close()

    ''' ANALYTICAL SOLUTION '''

    def euler(self):
        v, r, s = [np.mean(self.v0)], [self.r0], [self.s0]
        for i in range(1, int(self.T/self.STEP)):
            r.append(r[i-1] + self.STEP*(self.delta/(self.tau_m*math.pi) + 2*r[i-1]*v[i-1] - self.g*r[i-1] - 2*self.tau_m*math.log(self.a)*r[i-1]**2)/self.tau_m)
            v.append(v[i-1] + self.STEP*(v[i-1]**2 + self.eta_mean + self.J*self.tau_m*s[i-1] + self.I[i-1] - (math.log(self.a)**2 + math.pi**2)*(self.tau_m*r[i-1])**2 + self.delta*math.log(self.a)/math.pi )/self.tau_m)
            s.append(s[i-1] + self.STEP*(-s[i-1] + r[i-1])/self.tau_d)
        return np.array(v), np.array(r), np.array(s)

    def R(self, z) :
        return (1/(math.pi*self.tau_m))*((1-z.conjugate())/(1+z.conjugate())).real

    def V(self, z) :
        return ((1-z.conjugate())/(1+z.conjugate())).imag + self.tau_m*math.log(self.a)*self.R(z)

    def dF(self, z, s, I):
        return (1j/2) * ((z+1)**2) * (self.eta_mean + self.J*self.tau_m*s + self.g*self.V(z) + I) - ((z+1)**2)*self.delta/2 - ((1-z)**2)*(1j/2) + (1-z**2)*self.g/2

    def euler_kuramoto(self):
        w = math.pi*self.tau_m*self.r0 + 1j *(np.mean(self.v0) - self.tau_m*math.log(self.a)*self.r0)
        z = [(1-w.conjugate())/(1+w.conjugate())]
        s = [self.s0]
        for i in range(1, int(self.T/self.STEP)) :
            z.append(z[i-1] + self.STEP*self.dF(z[i-1], s[i-1], self.I[i-1]))
            s.append(s[i-1] + self.STEP*(-s[i-1] + self.R(z[i-1]))/self.tau_d)
        z = np.array(z)
        return np.absolute(z)


    ''' PLOTTING '''

    def plot_system(self, analytical=True) :
        # Generating subplots
        fig, ax = plt.subplots(5, 1, figsize=(12,7), sharex=True)

        # Retrieve simulation solutions
        self.retrieve_files()

        # Simulation solutions
        # Plotting injected current I
        times = [float(self.STEP*k) for k in range(len(self.input))]
        ax[0].plot(times, self.input, color='black')
        ax[0].set_ylabel(r'$I(t)$')
        ax[0].set_ylim(-0.5, 3)
        ax[0].set_yticks([0, 2.5], [r'$0$', r'$2.5$'])

        # Plotting voltage average
        times_v = [float(self.STEP*k) for k in range(len(self.v))]
        ax[1].plot(times_v, self.v, c='k', label='simulation')
        ax[1].set_ylim(-9)
        ax[1].set_yticks([-9, 9], [r'$-9$', r'$9$'])
        ax[1].set_ylabel(r'$v_{avg}(t)$')

        # Plotting firing rate average
        times = [float(self.STEP*k) for k in range(len(self.r))]
        ax[2].plot(times, self.r, c='k')
        ax[2].set_ylabel(r'$r(t)$')

        # Plotting synatic activation average
        times = [float(self.STEP*k) for k in range(len(self.s))]
        ax[3].plot(times, self.s, c='k')
        ax[3].set_ylabel(r'$s(t)$')

        # Plotting Kuramoto order parameter abs value
        #times = [float(self.STEP*k) for k in range(len(self.z))]
        #ax[4].plot(times, self.z, c='k')
        #ax[4].set_ylim(-0.1, 1.1)
        #ax[4].set_ylabel(r'$|Z(t)|$')

        # Plotting raster plot
        ax[4].set_ylabel('neuron #')
        ax[4].scatter(self.times_raster, self.raster, s=0.9, alpha=0.25, c='k')
        ax[4].set_ylim(0, self.selected_neurons)
        ax[4].set_yticks([0, 10000], [r'$0$', r'$1e4$'])
        ax[4].set_xlabel('time')
        ax[4].set_xlim(0, self.T)


        # Analytical solution
        if analytical :
            v_sol, r_sol, s_sol = self.euler()
            ax[1].plot([self.STEP*k for k in range(len(v_sol))], v_sol, c='r', label='analytical')
            ax[2].plot([self.STEP*k for k in range(len(r_sol))], r_sol, c='r')
            ax[3].plot([self.STEP*k for k in range(len(s_sol))], s_sol, c='r')

            #z_sol = self.euler_kuramoto()
            #ax[4].plot([float(self.STEP*k) for k in range(len(z_sol))], z_sol, c='r')

            # Computing frequency
            peaks, _ = find_peaks(r_sol, height=0.1)
            print(len(peaks)*10, 'Hz')

        #ax[1].legend(loc='upper right')

        #plt.suptitle('Network of 10e4 electrically and chemically coupled neurons')
        plt.tight_layout()
        fig.align_ylabels(ax)
        plt.subplots_adjust(hspace=0.5)

        plt.savefig(f'{self.subfolder}/full_network.png', dpi=600)

        plt.show()
        plt.close()
