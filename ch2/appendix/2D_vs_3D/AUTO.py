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

fig, ax = plt.subplots(1, 2, figsize=(12,5), sharey=True)

plt.rcParams['axes.xmargin'] = 0

color_list = ['#aa3863', '#31688e', '#35b779', '#fde725']

hb2Dm = auto.loadbd('hb_2Dm')
hb2Dp = auto.loadbd('hb_2Dp')
lp12D = auto.loadbd('lp1_2D')
lp22D = auto.loadbd('lp2_2D')

hb3Dm = auto.loadbd('hb_3Dm')
hb3Dp = auto.loadbd('hb_3Dp')
lp13D = auto.loadbd('lp1_3D')
lp23D = auto.loadbd('lp2_3D')

ax[0].set_ylabel(r'chemical coupling $J$')
ax[0].set_xlabel(r'electrical coupling $g$')
ax[1].set_xlabel(r'electrical coupling $g$')

line_w = 2.2

# HOPF BIFURCATION
ax[0].plot(hb2Dm['G'], hb2Dm['J'], color=color_list[0], linewidth=line_w, linestyle='-', zorder=10, label='HB')
ax[0].plot(hb2Dp['G'], hb2Dp['J'], color=color_list[0], linewidth=line_w, linestyle='-', zorder=10)
ax[0].plot(lp12D['G'], lp12D['J'], color=color_list[1], linewidth=line_w, linestyle='-', label='SN')
ax[0].plot(lp22D['G'], lp22D['J'], color=color_list[1], linewidth=line_w, linestyle='-')
ax[0].scatter(hb2Dp['G'][-1], hb2Dp['J'][-1], color=color_list[0], zorder=10, label='TB')

ax[1].plot(hb3Dm['G'], hb3Dm['J'], color=color_list[0], linewidth=line_w, linestyle='-', zorder=10, label='HB')
ax[1].plot(hb3Dp['G'], hb3Dp['J'], color=color_list[0], linewidth=line_w, linestyle='-', zorder=10)
ax[1].plot(lp13D['G'], lp13D['J'], color=color_list[1], linewidth=line_w, linestyle='-', label='SN')
ax[1].plot(lp23D['G'], lp23D['J'], color=color_list[1], linewidth=line_w, linestyle='-')
ax[1].scatter(hb3Dm['G'][-1], hb3Dm['J'][-1], color=color_list[0], zorder=10, label='TB')

ax[0].set_xlim(0,5)
ax[1].set_xlim(0,5)
ax[0].set_ylim(-50,10)

ax[0].legend(loc='upper right', fontsize=14)
ax[1].legend(loc='upper right', fontsize=14)

ax[0].set_title('2D model')
ax[1].set_title('3D model')

plt.tight_layout()
plt.savefig('2D_vs_3D.png', dpi=300)
plt.show()