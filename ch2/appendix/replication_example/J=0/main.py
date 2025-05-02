import Network as Ntw
import numpy as np
import math as math

# Parameters
T = 120
STEP = 1e-3

N = 10**4

VR = -100
VP = 100

g = 3 #5 #.5 #4.9 #1
J = 0 #-5 #-15 #-10

tau_m = 10
tau_s = 2.5*0.01

REFRACT_TIME = tau_m/(VP)

eta_mean = 1
delta = 1

# Initial states
r0 = 0
v0 = np.zeros(N)

# Initialize network
network = Ntw.QIFNetwork(T, N, tau_m, tau_s, J, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, r0)

# Run program
network.run()

# Plots
QIF_plots = network.plot_system(analytical=True)
