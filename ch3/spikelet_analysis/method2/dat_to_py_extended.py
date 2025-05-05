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

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

plt.figure(figsize=(8,4))

plt.rcParams['axes.xmargin'] = 0
colors = ['#493548', '#4B4E6D', '#6A8D92', '#80B192', '#A1E887']

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

colors = plt.cm.viridis(np.linspace(0,.8,len(args.files)))
colors = colors[::-1]
#label = ['stable', 'unstable', 'limit cycle', 'limit cycle']
label=['$b=1$', '$b=0.02$', '$b=0.01$']
s = ['-', '--', '-', '--']
j = -1

for filename in args.files :
    x, y = [], []
    y1, y2 = [], []
    x1, x2 = [], []
    k = []
    j += 1
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')
        last_line_nb = row_count(filename)

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:
            if datareader.line_num == 1 :
                k.append(int(row[3]))

            if datareader.line_num == row_count(filename) :
                x2.append(float(row[0]))
                y2.append(float(row[2]))
                x1.append(float(row[0]))
                y1.append(float(row[1]))

            if k[-1] != int(row[3]) or datareader.line_num == row_count(filename) :
                
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

                if len(x2) == 0:
                    k.append(int(row[3]))
        
            if float(row[1]) != float(row[2]) :
                x2.append(float(row[0]))
                y2.append(float(row[2]))

            x1.append(float(row[0]))
            y1.append(float(row[1]))


    for i in range(len(x)) :
        if k[i]-1 == 0:
            plt.plot(x[i], y[i], linestyle=s[k[i]-1], color=colors[j], label=label[j])
        else:
            plt.plot(x[i], y[i], linestyle=s[k[i]-1], color=colors[j])

plt.xlabel(r'electrical coupling $g$')
plt.ylabel(r'network firing rate $r$')

plt.xlim(0, 10)
plt.ylim(-.25, 6)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.savefig('bif_spikelets.png', dpi=600)
plt.show()
