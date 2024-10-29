#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs='+')
parser.add_argument('--t0', type=float)
parser.add_argument('--labels', nargs='*')
args = parser.parse_args()

if len(args.labels) > 0:
    assert len(args.labels) == len(args.datafiles)

for fname in args.datafiles:
    
    ang, N, tau = np.loadtxt(fname, comments='#')
    t = 0.0
    with open(fname, 'r') as f:
        line = '#'
        for line in f:
            if not line.startswith('#'):
                break
            if line.contains("time"):
                t = float(line.split(':')[1].strip())
    
    if len(args.labels) == 0:
        label = f"t + t0 = {t:.1f}"
    else:
        label = args.labels[i]
    plt.plot(ang*180/np.pi, N * (1. + t/t0)**2, label=label)

plt.xlabel(r'$\theta~(^{\circ})$')
plt.ylabel(r'$N (1 + t/t_0)^2~(\mathrm{cm^{-2}})$')
plt.yscale("log")
plt.legend()
plt.gcf().savefig("N_vs_theta.png", dpi=480)
