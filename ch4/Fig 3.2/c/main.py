import Network as Ntw
import numpy as np

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

# Initialize network
network = Ntw.QIFNetwork(T, N, tau_m, tau_d, tau_s1, tau_s2, I, Js, Jc, g, eta_mean, delta, VP, VR, STEP, v0, s0, r0)

# Run program
network.run()

# Plots
QIF_plots = network.plot_system(analytical=False)
