import Network as Ntw
import numpy as np
import math as math

# Parameters
T = 100
STEP = 1e-3

N = 10**4

VR = -100
VP = 100

g = 0.37
J = 9.5

tau_m = 1
tau_s = 0.0182

REFRACT_TIME = tau_m/(VP)

eta_mean = -1
delta = 0.3

I = np.concatenate([np.zeros(int(T/(STEP*2))), 1.2*np.ones(int(T/(STEP*90))), np.zeros(int(T/(STEP*2)))], axis=0)

# Initial states
r0 = 0.
v0 = 0*np.ones(N)

# Initialize network
network = Ntw.QIFNetwork(T, N, tau_m, tau_s, J, g, eta_mean, I, delta, VP, VR, REFRACT_TIME, STEP, v0, r0)

# Run program
network.run()

# Plots
QIF_plots = network.plot_system(analytical=True)
