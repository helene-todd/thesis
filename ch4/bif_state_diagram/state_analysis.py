import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from scipy.optimize import fsolve
from numpy.linalg import eig
import math as math
import cmath as cmath
import numpy as np
import os, csv
from pathlib import Path

# General parameters
T = 1000
STEP = 1e-3
tau_m = 1
tau_d = 1
eta = 1
delta = 0.3
Jc = -8

def euler(r0, v0, s0, g, J) :
    v, r, s = [[np.mean(np.random.normal(v0[0], 1, 400))], [np.mean(np.random.normal(v0[1], 1, 1000))]], [[r0[0]], [r0[1]]], [[s0[0]], [s0[1]]]
    for i in range(1, int(T/STEP)):
        r[0].append(r[0][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[0][i-1]*v[0][i-1] - g[0]*r[0][i-1])/tau_m)
        r[1].append(r[1][i-1] + STEP*(delta/(tau_m*math.pi) + 2*r[1][i-1]*v[1][i-1] - g[1]*r[1][i-1])/tau_m)
        v[0].append(v[0][i-1] + STEP*(v[0][i-1]**2 + eta + J[0]*tau_m*s[0][i-1] + Jc*tau_m*s[1][i-1] - (math.pi**2)*(tau_m*r[0][i-1])**2)/tau_m)
        v[1].append(v[1][i-1] + STEP*(v[1][i-1]**2 + eta + J[1]*tau_m*s[1][i-1] + Jc*tau_m*s[0][i-1] - (math.pi**2)*(tau_m*r[1][i-1])**2)/tau_m)
        s[0].append(s[0][i-1] + STEP*(-s[0][i-1] + r[0][i-1])/tau_d)
        s[1].append(s[1][i-1] + STEP*(-s[1][i-1] + r[1][i-1])/tau_d)
    return np.array(r[0][int(2*T/(3*STEP)):]), np.array(v[0][int(2*T/(3*STEP)):]), np.array(s[0][int(2*T/(3*STEP)):])

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

def get_equilibria(g, J) :
    # Computing equilibria
    def f(r):
        eq1 = delta**2/(2*tau_m*math.pi)**2 + (g[0]*r[0])**2/4 - delta*g[0]*r[0]/(2*tau_m*math.pi) + eta*r[0]**2 + J[0]*tau_d*r[0]**3 + Jc*tau_d*r[1]*r[0]**2 - (math.pi*tau_m*r[0]**2)**2
        eq2 = delta**2/(2*tau_m*math.pi)**2 + (g[1]*r[1])**2/4 - delta*g[1]*r[1]/(2*tau_m*math.pi) + eta*r[1]**2 + J[1]*tau_d*r[1]**3 + Jc*tau_d*r[0]*r[1]**2 - (math.pi*tau_m*r[1]**2)**2
        return np.array([eq1, eq2])

    r2 = np.linspace(0.01, 1, 1000)
    r1 = -delta**2/(Jc*tau_d*r2**2*4*tau_m**2*math.pi**2) - g[1]**2/(4*Jc*tau_d) + delta*g[1]/(Jc*tau_d*r2*2*tau_m*math.pi) - eta/(Jc*tau_d) - J[1]*tau_d*r2/(Jc*tau_d) + math.pi**2*tau_m*r2**2/(Jc*tau_d)
    y = delta**2/(2*tau_m*math.pi)**2 + (g[0]*r1)**2/4 - delta*g[0]*r1/(2*tau_m*math.pi) + eta*r1**2 + J[0]*tau_d*r1**3 + Jc*tau_d*r2*r1**2 - (math.pi*tau_m*r1**2)**2
    idx = np.argwhere(np.diff(np.sign(y))).flatten()

    r_e, v_e, s_e = [], [], []

    for el in idx :
        if r1[el] > 0 :
            r_e.append(fsolve(f, x0=[r1[el], r2[el]]))
            v_e.append([-delta/(2*r_e[-1][0]*tau_m*math.pi) + g[0]/2, -delta/(2*r_e[-1][1]*tau_m*math.pi) + g[1]/2])
            s_e.append([r_e[-1][0], r_e[-1][1]])

    return np.array(r_e), np.array(v_e), np.array(s_e)

def euclidean_distance(x, y):
    res = 0
    for i in range(len(x)) :
        res += (x[i]-y[i])**2
    return math.sqrt(res)

def get_eigenvalues_classification(g, J) :
    
    r_e, v_e, s_e = get_equilibria(g, J)

    all_eigenvalues_classification = []
    # Computing eigenvalues
    for k in range(len(r_e)):
        a = np.array([ [2*v_e[k][0]-g[0], 0, 2*r_e[k][0], 0, 0, 0 ],
                       [0, 2*v_e[k][1]-g[1], 0, 2*r_e[k][1], 0, 0 ],
                       [-2*math.pi**2*tau_m**2*r_e[k][0], 0, 2*v_e[k][0], 0, J[0]*tau_d, Jc*tau_d ],
                       [0, -2*math.pi**2*tau_m**2*r_e[k][1], 0, 2*v_e[k][1], Jc*tau_d, J[1]*tau_d ],
                       [1, 0, 0, 0, -1, 0 ],
                       [0, 1, 0, 0, 0, -1 ] ])
        w,v=eig(a)
        #print(f'Equilibria : \nr = {r_e[k]}\nv = {v_e[k]}\ns = {s_e[k]}\n\n', f'Eigenvalues: {w},\nEigenvectors: {v}')

        eigenvalues_classification = np.zeros(shape=(len(w), 3))
        for k in range(len(w)) :
            if np.real(w[k]) > 0 :
                eigenvalues_classification[k][0] = 1
            if np.real(w[k]) < 0 :
                eigenvalues_classification[k][1] = 1
            if np.imag(w[k]) != 0 :
                eigenvalues_classification[k][2] = 1

        all_eigenvalues_classification.append(eigenvalues_classification)

    return all_eigenvalues_classification

# 1 : LC + SS
# 2 : LC + LC
# 3 : SS
# 4 : LC

x, y = np.linspace(0,1.3,50), np.linspace(-10,0,100) #(g1,Js2)
X, Y = np.meshgrid(x, y)

my_file = Path('pop_overlay.txt')
if my_file.is_file():
    pop_overlay = []
    with open('pop_overlay.txt') as f:
        for line in f:
            inner_list = np.array([elt.strip() for elt in line.split(' ')[:-1]], dtype=int)
            pop_overlay.append(inner_list)
    pop_overlay = np.array(pop_overlay)
else :
    pop_overlay = np.zeros_like(X.T, dtype=int)
    f = open("pop_overlay.txt", "a")
    for i in range(len(x)) :
        print(i,'/',len(x))
        for j in range(len(y)) :
            g = [x[i], 2]
            J = [-2.5, y[j]]
            #print(g, J)
            output = np.array(get_eigenvalues_classification(g, J))
            # case one equilibrium (trivial)
            if len(output) == 1 :
                if [1, 0, 1] in output[0].tolist() :
                    pop_overlay[i,j] = 4
                else :
                    pop_overlay[i,j] = 3
            # case three equilibria
            if len(output) == 3 :
                if ([1, 0, 1] in output[0].tolist()) and ([1, 0, 1] in output[2].tolist()) : # 2 limit cycles present: is one unstable?
                    ''' Determines if in state 2 or 4 '''
                    epsilon = 0.01
                    r_e, v_e, s_e = get_equilibria(g, J)
                    r1, v1, s1 = euler(np.round(r_e[0],5), np.round(v_e[0],5), np.round(s_e[0],5), g, J) #the first equilibrium is the unstable LC
                    r2, v2, s2 = euler(np.round(r_e[2],5), np.round(v_e[2],5), np.round(s_e[2],5), g, J) #the third equilibrium is the stable LC
                    if abs(np.mean(r2[-int(len(r2)/2):]) - np.mean(r1[-int(len(r1)/2):])) < epsilon: #unstable limit cycle converges to stable LC
                        pop_overlay[i,j] = 4
                    else :
                        pop_overlay[i,j] = 2
                
                if ([1, 0, 1] in output[0].tolist()) ^ ([1, 0, 1] in output[2].tolist()) : # 1 limit cycle present + another equilibrium: is it unstable?
                    ''' Determines if in state 1 or 3 '''
                    epsilon = 0.005
                    r_e, v_e, s_e = get_equilibria(g, J)
                    r1, v1, s1 = euler(np.round(r_e[2],5), np.round(v_e[2],5), np.round(s_e[2],5), g, J) #the third equilibrium is the LC
                    print(r1)
                    # check for convergence of LC to SF equilibrium
                    if abs(r1[-1] - r_e[0][0]) <  epsilon : # if the variable r (for example, sufficient) converges to r_eq of SF
                        pop_overlay[i,j] = 3 # LC is unstable (and SF is stable)
                    else :
                        pop_overlay[i,j] = 1 # LC is stable (and SF is stable)
            f.write(f'{pop_overlay[i,j]} ')
        f.write('\n')

''' PLOTTING '''
cmp = ListedColormap(['#aa3863', '#31688e', '#35b779', '#fde725'])
plt.pcolormesh(X, Y, pop_overlay.T, shading='auto', cmap=cmp)
f.close()

label = ['bifurcation']
s = ['-', '--', ':']
pop_col = ['k', '#119a7b']
p = 0

x, y = [], []
y1, y2 = [], []
x1, x2 = [], []
k = []

plt.ylabel(r'chemical coupling $J_2$')
plt.xlabel(r'electrical coupling $g_1$')

plt.xlim(0,1.3)
plt.ylim(-10,0)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), loc='lower right')
#plt.colorbar()

plt.tight_layout()
plt.savefig('pop_bif_diagram.png', dpi=600)
plt.show()