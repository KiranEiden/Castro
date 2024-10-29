#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs='+')
parser.add_argument('--teng', type=float, default=2047.49826674)
parser.add_argument('--labels', nargs='*')
args = parser.parse_args()

if args.labels is not None:
    assert len(args.labels) == len(args.datafiles)

for i, fname in enumerate(args.datafiles):
    
    ang, N, tau = np.loadtxt(fname, comments='#').T
    t = 0.0
    with open(fname, 'r') as f:
        line = '#'
        for line in f:
            if not line.startswith('#'):
                break
            if "time" in line:
                t = float(line.split(':')[1].strip())
    
    if args.labels is None:
        label = f"t = {t/args.teng:.1f}" + r"$t_{\mathrm{eng}}$"
    else:
        label = args.labels[i]
    plt.plot(ang*180/np.pi, N * (0.1 + t/args.teng)**2, label=label)

plt.xlabel(r'$\theta~(^{\circ})$')
plt.ylabel(r'$N (\tilde{t}_0 + \tilde{t})^2~(\mathrm{cm^{-2}})$')
plt.yscale("log")
plt.legend()
plt.gcf().savefig("N_vs_theta.png", dpi=480)
