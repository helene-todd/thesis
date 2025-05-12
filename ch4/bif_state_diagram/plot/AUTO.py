import auto
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

# Make sure to run 'modelclean.auto' (located in AUTO folder), then copy/paste output s. and b. files to running directory

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

plt.figure(figsize=(10.5,7.5))

plt.rcParams['axes.xmargin'] = 0

color_list = ['#aa3863', '#31688e', '#35b779', '#fde725']
#line_colors = plt.cm.gray(np.linspace(0,1,5))

h1 = auto.loadbd('hbrg1j2ref')
h2 = auto.loadbd('hbrg1j2ref2')
h1m = auto.loadbd('hbrg1j2refm')

lp1 = auto.loadbd('lp2j2g1')
lp2 = auto.loadbd('lp2j2g1_2')

pd1 = auto.loadbd('pd2parg1j2go')

snpp1 = auto.loadbd('snp1per_J2G1_go')
snpp1m = auto.loadbd('snp1per_J2G1_gom')

tr = auto.loadbd('TR')

plt.ylabel(r'chemical coupling $J_2$')
plt.xlabel(r'electrical coupling $g_1$')

plt.xlim(0,1.3)
plt.ylim(-10,0)

line_w = 2.4

# HOPF BIFURCATION
plt.plot(h1['G1'][h1['G1']<=0.72], h1['J2'][h1['G1']<=0.72], color='k', linewidth=line_w, linestyle='-')
plt.plot(h1['G1'][h1['G1']>0.72], h1['J2'][h1['G1']>0.72], color='grey', linewidth=line_w, linestyle='-')
plt.plot(h2['G1'], h2['J2'], color='grey', linewidth=line_w, linestyle='-')
plt.plot(h1m['G1'], h1m['J2'], color='k', linewidth=line_w, linestyle='-')

# SADDLE NODE OF EQUILIBRIA
plt.plot(lp1['G1'], lp1['J2'], color='grey', linewidth=line_w, linestyle=(0, (3, 5, 1, 5)))
plt.plot(lp2['G1'], lp2['J2'], color='grey', linewidth=line_w, linestyle=(0, (3, 5, 1, 5)))

# PERIOD DOUBLING
plt.plot(pd1['G1'][:15], pd1['J2'][:15], color='grey', linewidth=line_w, linestyle=':')
plt.plot(pd1['G1'][15:45], pd1['J2'][15:45], color='k', linewidth=line_w, linestyle=':')
plt.plot(pd1['G1'][45:], pd1['J2'][45:], color='grey', linewidth=line_w, linestyle=':')

# TORUS RESONANCE BIFURCATION
plt.plot(tr['G1'], tr['J2'], color='k', linewidth=line_w, linestyle='--')

# SADDLE NODE OF CYCLES
plt.plot(snpp1['G1'], snpp1['J2'], color='k', linewidth=line_w, linestyle='-.')
plt.plot(snpp1m['G1'][snpp1m['G1']< 0.92], snpp1m['J2'][snpp1m['G1']< 0.92], color='grey', linewidth=line_w, linestyle='-.')
plt.plot(snpp1m['G1'][snpp1m['G1']>= 0.92], snpp1m['J2'][snpp1m['G1']>= 0.92], color='k', linewidth=line_w, linestyle='-.')


x, y = np.linspace(0,1.3,100), np.linspace(-10,0,100) #(g1,Js2)
X, Y = np.meshgrid(x, y)

my_file = Path('pop_overlay.txt')
if my_file.is_file():
    pop_overlay = []
    with open('pop_overlay.txt') as f:
        for line in f:
            inner_list = np.array([elt.strip() for elt in line.split(' ')[:-1]], dtype=int)
            pop_overlay.append(inner_list)
    pop_overlay = np.array(pop_overlay)

cmp = ListedColormap(['#aa3863', '#31688e', '#35b779', '#fde725'])
plt.pcolormesh(X, Y, pop_overlay.T, shading='auto', alpha=.75, cmap=cmp)
f.close()

plt.ylabel(r'chemical coupling $J_2$')
plt.xlabel(r'electrical coupling $g_1$')

plt.xlim(0,1.3)
plt.ylim(-8,0)

plt.tight_layout()
plt.savefig('pop_bif_diagram.png', dpi=600)
plt.show()