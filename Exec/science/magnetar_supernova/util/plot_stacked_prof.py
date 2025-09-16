#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
import analysis_util as au

prop_cycle = plt.rcParams['axes.prop_cycle']
mpl_colors = prop_cycle.by_key()['color']

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument("--nplot", type=int, default=100)
parser.add_argument('-t0', "--time_offset", type=float)
parser.add_argument('--teng', type=float, default=2047.49826674)
parser.add_argument('-f', '--fields', default=[r'density:$\rho~(\mathrm{g \cdot cm^{-3}})$',
        r'x_velocity:$u_{r}~(\mathrm{cm/s})$', r'pressure:$P~(\mathrm{dyn \cdot cm^{-2}})$'])
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
ts = map(tf, ts)

def plot_avg_prof(ds, fig, ax):
    
    if args.time_offset is not None:
        xlabel = r"$R/t$ (cm/s)"
    else:
        xlabel = r"$R$ (cm)"
    
    for i, flpair in enumerate(args.fields):
        
        field, ylabel = flpair.split(':')
        print(f"Plotting angle-averaged {field} profile for {ds}.")
        
        avg, r1d = au.get_avg_prof_2d(ds, args.nplot, field=field, level=args.level, return_r=True)
        
        if args.time_offset is not None:
            x = r1d / (ds.current_time.d + args.time_offset)
        else:
            x = r1d
        
        ax[i].plot(x, avg, label=f"t = {ds.current_time.d/args.teng:.2f}" + r"$~\mathrm{t_{eng}}$")
        
        ax[i].set_ylabel(ylabel)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
    
    plt.xlabel(xlabel)

fig, ax = plt.subplots(len(args.fields), 1, sharex='col')
fig.set_size_inches((8, 12))
    
for ds in ts:
    plot_avg_prof(ds, fig, ax)

ax[-1].legend()
fig.savefig("stacked_profiles.pdf", dpi=480)
