from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os
import csv
import argparse

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

plt.figure(figsize=(6,4.5))

# TO DO : Rewrite this code to make it more readable.
# USAGE : Run in terminal  "python dat_to_py.py *.dat"

plt.rcParams['axes.xmargin'] = 0
colors = ['#493548', '#4B4E6D', '#6A8D92', '#80B192', '#A1E887']

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

colors = plt.cm.viridis(np.linspace(0,1,len(args.files)))
s = ['-', '--', ':']

col = 0

for filename in args.files :
    g = []
    b = []
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:

            g.append(float(row[0]))
            b.append(float(row[1]))

        plt.plot(g, b, color='k', linestyle='--', label='HB point')
       

#plt.title('Bifurcation Diagram', size=18)
plt.xlabel(r'electrical coupling $g$')
plt.ylabel(r'depolarisation amplitude $b$')

plt.xlim(0,2)
plt.ylim(0,1)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right') # bbox_to_anchor=(1, 0.95))

#plt.title('Bifurcation Diagram', size=14) #, pad=20)
plt.tight_layout()
plt.savefig('HB_constant.png', dpi=600)
plt.show()
