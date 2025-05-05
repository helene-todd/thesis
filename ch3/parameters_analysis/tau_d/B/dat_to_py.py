from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os
import csv
import argparse

# TO DO : Rewrite this code to make it more readable.
# USAGE : Run in terminal  "python dat_to_py.py *.dat"

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

plt.rcParams['axes.xmargin'] = 0
colors = ['#493548', '#4B4E6D', '#6A8D92', '#80B192', '#A1E887']

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

palette = plt.cm.plasma(np.linspace(0,0.75,4))
colors = ['k', 'k', '#40C1AC']

label = ['stable', 'unstable', 'limit cycle']
s = ['-', '--', '-']
x, y = [], []
y1, y2 = [], []
x1, x2 = [], []
k = []

for filename in args.files :
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')
        last_line_nb = row_count(filename)

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:
            if datareader.line_num == 1 :
                k.append(int(row[3]))

            if k[-1] != int(row[3]) :
                if int(row[3]) in [1, 2, 3] :
                    k.append(int(row[3]))
                if len(x1) > 1 :
                    x.append(x1)
                    x1 = []
                if len(x2) > 1 :
                    x.append(x2)
                    x2 = []
                    k.append(k[-1])
                if len(y1) > 1 :
                    y.append(y1)
                    y1 = []
                if len(y2) > 1 :
                    y.append(y2)
                    y2 = []

            if float(row[1]) != float(row[2]) or k[-1] == 3 :
                x2.append(float(row[0]))
                y2.append(float(row[2]))

            x1.append(float(row[0]))
            y1.append(float(row[1]))

for i in range(len(x)) :
    plt.plot(x[i], y[i], linestyle=s[k[i]-1], linewidth=2, color=colors[k[i]-1], label=label[k[i]-1])

plt.xlabel(r'synaptic time constant $\tau_d$')
plt.ylabel(r'network firing rate $r$')

plt.ylim(0,1)
plt.xlim(0,20)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.savefig('bif_taud.png', dpi=600)
plt.show()
