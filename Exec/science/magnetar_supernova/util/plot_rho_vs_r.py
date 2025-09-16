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
parser.add_argument('--no_avg', action='store_true')
parser.add_argument("--nplot", nargs=1, type=int)
parser.add_argument('-t0', "--time_offset", type=float)
parser.add_argument('--teng', type=float, default=2047.49826674)
parser.add_argument('--single_plot', action='store_true')
args = parser.parse_args()

if args.nplot is None:
    args.nplot = 5 if args.no_avg else 100

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
ts = map(tf, ts)

def plot_avg_rho_prof(ds):
    
    print(f"Plotting angle-averaged density profile for {ds}.")
    
    avg, r1d = au.get_avg_prof_2d(ds, args.nplot, field='density', level=args.level, return_r=True)
    
    if args.time_offset is not None:
        x = r1d / (ds.current_time.d + args.time_offset)
        xlabel = r"$R/t$ [cm/s]"
    elif args.time_offset is not None:
        x = r1d
        xlabel = r"$R$ [cm]"
    
    plt.plot(x, avg, label=f"t = {ds.current_time.d/args.teng:.2f}" + r"$~\mathrm{t_{eng}}$")
    
    plt.xlabel(xlabel)
    plt.ylabel(r"$\bar{\rho}$ [$\mathrm{g~cm^{-3}}$]")
    plt.xscale("log")
    plt.yscale("log")

for ds in ts:
    
    if args.no_avg:
        au.plot_prof_2d(ds, args.nplot, "density", ylabel=r"$\rho$ [$\mathrm{g~cm^{-3}}$]", log=True)
    else:
        plot_avg_rho_prof(ds)
        
    if not args.single_plot:
        plt.savefig(f"rho_prof_{ds}.png")
        plt.gcf().clear()

if args.single_plot:
    plt.legend()
    plt.savefig("rho_prof.png")
    plt.gcf().clear()
