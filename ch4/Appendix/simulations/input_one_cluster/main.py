import Network as Ntw
import numpy as np

# Parameters
T = 10
STEP = 1e-3

# Number of neurons in population 1, population 2
N = [10**4, 10**4]

VR = -100
VP = 100

# Electrical coupling strength in population 1, population 2
g = [0., 1.44]
# Chemical coupling strength in population 1, population 2
Js = [-1, 0.]
# Cross chemical coupling strength between the two populations
Jc = -6

tau_m = 2.8
tau_d = 2
tau_s1, tau_s2 = 2.*tau_m*1e-2, 2.*tau_m*1e-2 #1.215 * tau_m * 1e-2

REFRACT_TIME = tau_m/VP

eta_mean = 2
delta = .3

# Initialize input current vector
I1 = np.concatenate((np.zeros(int(T/(2*STEP))), -5*np.ones(int(T/(40*STEP))), np.zeros(int(T/(2*STEP)))), axis=0)
I2 = np.zeros(int(T/STEP))
print(len(I1), len(I2))
I = [I1, I2]

# Initial states in population 1, population 2
r0 = [0, 0]
v0 = [np.random.normal(0, 1, N[0]), np.random.normal(0, 1, N[1])]
s0 = [0, 0]

# Initialize network
network = Ntw.QIFNetwork(T, N, tau_m, tau_d, tau_s1, tau_s2, I, Js, Jc, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, s0, r0)

# Run program
network.run()

# Plots
QIF_plots = network.plot_system(analytical=False)
