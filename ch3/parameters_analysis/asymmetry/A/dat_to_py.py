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

colors = plt.cm.plasma(np.linspace(0,0.75,len(args.files)))
print(len(args.files))
s = ['-', '--', ':']
a = [0.5, '10^0', '10^1', '10^2', '10^3']

col = 0

for filename in args.files :
    g = [[]]
    J = [[]]
    stability = []
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')
        last_line_nb = row_count(filename)
        last_g = -99
        last_J = -99
        last_stability = 0

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:

            # the 2nd condition avoids a list with one value when two consecutive values are duplicates
            if last_g == float(row[0]) and len(g[-1]) > 1 :
                if last_stability != int(row[3]):
                    g[-1].append(last_g)
                    J[-1].append(last_J)
                g.append([])
                J.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            if last_g != -99 :
                g[-1].append(last_g)
                J[-1].append(last_J)

            if (last_stability != int(row[3]) and len(g[-1]) > 1) :#or (len(g[-1])>0 and float(row[0])-g[-1][-1] < 0):
                g.append([])
                J.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            # if at last line, then stop checking for consecutive values and just add the remaining data
            if last_line_nb == datareader.line_num:
                g[-1].append(float(row[0]))
                J[-1].append(float(row[1]))
                stability.append(int(row[3]))

            last_g = float(row[0])
            last_J = float(row[1])
            last_stability = int(row[3])

    for k in range(len(g)) :
        if stability[k] == 1 :
            plt.plot(g[k], J[k], color=colors[col], linestyle=s[0], label=f'$a={a[col]}$')
        if stability[k] == 2 :
            plt.plot(g[k], J[k], color=colors[col], linestyle=s[1], label=f'$a={a[col]}$')
        if stability[k] == 3 :
            plt.plot(g[k], J[k], color=colors[col], linestyle=s[2], label=f'$a={a[col]}$')
    col += 1

#plt.title('Bifurcation Diagram', fontsize=11)
plt.xlabel(r'electrical coupling $g$')
plt.ylabel(r'chemical coupling $J$')

plt.xlim(0, 3.5)
plt.ylim(-20,0)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right') # bbox_to_anchor=(1, 0.95))

#plt.title('Bifurcations for different reset values', size=18, pad=20)

plt.tight_layout()
plt.savefig('bif_a.png', dpi=600)
plt.show()
