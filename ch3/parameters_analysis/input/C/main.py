import Network as Ntw
import numpy as np

# For a=0.5, vr=-33, vp=46, g=1.2, J=-5, tau_s=1.215*tau_m*1e-2, eta_mea=1, delta=0.3

# Parameters
#T = 140
T = 90
STEP = 1e-3

N = 10**4

VR = -100
VP = 100

g = 1 #4.9 #1
J = -10 #-15 #-10

tau_m = 1
tau_d = 1
tau_s = 1.215 * tau_m * 1e-2 #1.215

REFRACT_TIME = tau_m/VP

eta_mean = 1
delta = 0.3

# Initialize input current vector
I = np.concatenate((np.zeros(int(T/(3*STEP))), .75*np.ones(int(T/(3*STEP))), 2.5*np.ones(int(T/(3*STEP))+1)), axis=0)
#I = np.zeros(int(T/STEP))

# Initial states
r0 = 0.
v0 = np.random.normal(-10, 1, N)
s0 = 0.

# Initialize network
network = Ntw.QIFNetwork(T, N, tau_m, tau_d, tau_s, I, J, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, s0, r0)

# Run program
network.run()

# Plots
QIF_plots = network.plot_system(analytical=True)
