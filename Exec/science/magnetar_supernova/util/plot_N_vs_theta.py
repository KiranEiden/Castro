#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs='+')
parser.add_argument('--teng', type=float, default=2047.49826674)
parser.add_argument('--plot_half', action='store_true')
parser.add_argument('--stack', action='store_true')
parser.add_argument('--labels', nargs='*')
args = parser.parse_args()

if args.labels is not None:
    assert len(args.labels) == len(args.datafiles)
    
if args.stack and len(args.datafiles) > 1:
    fig, ax = plt.subplots(len(args.datafiles), 1, sharex='col', sharey='col')
else:
    fig = plt.gcf()
    ax = [plt.gca()]

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
        label = f"t = {t/args.teng:.2f}" + r"$t_{\mathrm{eng}}$"
    else:
        label = args.labels[i]

    
    if args.plot_half:
        mask = ang <= np.pi/2.
        th_plot = ang[mask]
        N_plot = N[mask]
    else:
        th_plot = ang
        N_plot = N
    ax[i % len(ax)].plot(th_plot*180./np.pi, N_plot / N_plot.mean(), label=label, linewidth=1, color=colors[i])

plt.xlabel(r'$\theta~(^{\circ})$')
for axis in ax:
    axis.set_ylabel(r'$N / \langle N \rangle$')
    axis.set_yscale("log")
    axis.legend()
fig.savefig("N_vs_theta.png", dpi=480)
