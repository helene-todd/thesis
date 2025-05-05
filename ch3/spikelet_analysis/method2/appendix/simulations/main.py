import Network as Ntw
import numpy as np

# Parameters
T = 200
STEP = 1e-3

N = 10**4

VR = -100
VP = 100

g = 3.75
J = -4

tau_m = 1
tau_d = 1
tau_s = 1.95 * tau_m * 1e-2 #1.5

REFRACT_TIME = 3*tau_m/VP # 4, seed 2

eta_mean = 1
delta = 0.3

# Initialize input current vector
I = np.concatenate((np.zeros(int(T/(2*STEP))), np.zeros(int(T/(4*STEP))), 2*np.ones(int(T/(200*STEP))), np.zeros(int(T/(STEP)))), axis=0)

# Initial states
r0 = 0.2681062
v0 = 1.32165
s0 = 0.2681062

# Initialize network
network = Ntw.QIFNetwork(T, N, tau_m, tau_d, tau_s, I, J, g, eta_mean, delta, VP, VR, REFRACT_TIME, STEP, v0, s0, r0)

# Run program
network.run()

# Plots
QIF_plots = network.plot_system()
