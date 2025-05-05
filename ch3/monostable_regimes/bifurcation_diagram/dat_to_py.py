from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os, csv, argparse
from scipy.signal import find_peaks
from pathlib import Path

plt.rcParams['axes.xmargin'] = 0
colors = ['#493548', '#4B4E6D', '#6A8D92', '#80B192', '#A1E887']


SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=16)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

s = ['-', '--', ':']

''' PARAMETERS '''
T = 300
STEP = 1e-3
N = 10**2
VR = -100
VP = 100
a = abs(VP/VR)
tau_m = 1
tau_d = 1
tau_s = 1.215 * tau_m * 1e-2
REFRACT_TIME = tau_m/VP
eta_mean = 1
delta = .3

# Initialize input current vector
I = np.zeros(int(T/STEP))

# Initial states
r0 = 0
v0 = np.random.normal(0, 1, N)
s0 = 1

''' ANALYTICAL SOLUTION '''
def euler(g, J):
    v, r, s = [np.mean(v0)], [r0], [s0]
    for i in range(1, int(T/STEP)):
        r.append(r[i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[i-1]*v[i-1] - g*r[i-1] - 2*tau_m*math.log(a)*r[i-1]**2)/tau_m)
        v.append(v[i-1] + STEP*(v[i-1]**2 + eta_mean +J*tau_m*s[i-1] + I[i-1] - (math.log(a)**2 + math.pi**2)*(tau_m*r[i-1])**2 + delta*math.log(a)/math.pi )/tau_m)
        s.append(s[i-1] + STEP*(-s[i-1] + r[i-1])/tau_d)
    return np.array(r)

''' RETRIEVING DATA '''
for filename in args.files :
    g = []
    J = []
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')
        last_line_nb = row_count(filename)
        last_g = -99
        last_J = -99
        last_stability = 0

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:

            # the 2nd condition avoids a list with one value when two consecutive values are duplicates
            if last_g == float(row[0]) and len(g) > 1 :
                if last_stability != int(row[3]):
                    g.append(last_g)
                    J.append(last_J)
                g.append([])
                J.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            if last_g != -99 :
                g.append(last_g)
                J.append(last_J)

            # if at last line, then stop checking for consecutive values and just add the remaining data
            if last_line_nb == datareader.line_num:
                g.append(float(row[0]))
                J.append(float(row[1]))

            last_g = float(row[0])
            last_J = float(row[1])


''' FREQUENCY NUMERICAL COMPUTATIONS '''
# Creating meshgrid
x, y = np.linspace(0,3,100), J
X, Y = np.meshgrid(x, y)

my_file = Path('bif_overlay.txt')

if my_file.is_file():
    Fq = []
    with open('bif_overlay.txt') as f:
        for line in f:
            inner_list = np.array([elt.strip() for elt in line.split(' ')], dtype=float)
            Fq.append(inner_list)
    Fq = np.array(Fq)
else :
    Fq = np.empty((len(J),100,))
    Fq[:] = np.nan
    for j in range(len(y)) :
        print(j, '/', len(y))
        g_min = g[j]
        for i in range(len(x)) :
            if x[i] > g_min :
               r_sol = euler(x[i], y[j])
               # Computing frequency
               peaks, _ = find_peaks(r_sol, height=0.1)
               Fq[j,i] = len(peaks)*100/T
    with open('bif_overlay.txt', 'w') as f:
        csv.writer(f, delimiter=' ').writerows(Fq)


Fq[np.where(Fq < 5)] = np.nan

''' PLOTTING '''
# plot parameters
#plt.title('Bifurcation Diagram', fontsize=11)
plt.xlabel(r'electrical coupling $g$')
plt.ylabel(r'chemical coupling $J$')

plt.xlim(0, 3)
plt.ylim(-20,0)

# plot bifurcation line
plt.plot(g, J, color='k', linestyle=s[1], linewidth=4)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), loc='upper right') # bbox_to_anchor=(1, 0.95))

print(X.shape)
print(Y.shape)
print(Fq.shape)

minFq = min(Fq.flatten())
print(np.nanmin(Fq.flatten()))

plt.pcolormesh(X, Y, Fq, shading='auto')
plt.colorbar(label='spiking frequency (Hz)')

#plt.title('Spiking rate in oscillatory regime', size=18, pad=20)

plt.tight_layout()

plt.savefig('bifurcation_frequency.png', dpi=600)
plt.show()
