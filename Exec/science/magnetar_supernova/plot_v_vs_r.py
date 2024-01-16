#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from analysis_util import AMRData
from scipy.interpolate import RegularGridInterpolator

prop_cycle = plt.rcParams['axes.prop_cycle']
mpl_colors = prop_cycle.by_key()['color']

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('--plot_prof', action='store_true')
parser.add_argument('--plot_avg_prof', action='store_true')
parser.add_argument("--nplot", default=5)
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
ts = map(tf, ts)

def plot_prof(ds, fields, styles):
    
    print(f"Plotting profiles for {ds}.")
    
    rlo, zlo, plo = ds.domain_left_edge
    rhi, zhi, phi = ds.domain_right_edge
    zero = 0.0 * rlo
    
    for i, th in enumerate(np.linspace(0.0, np.pi, num=args.nplot)):
        
        ray = ds.ray((zero,)*3, (rhi*np.sin(th), rhi*np.cos(th), zero))
    
        r = np.sqrt(ray[('index', 'r')]**2 + ray[('index', 'z')]**2)
        idx = np.argsort(r)
    
        r = r[idx]
    
        for j, field in enumerate(fields):
        
            data = ray[field][idx]
            plt.plot(r, data, label=f"θ = {th*180/np.pi}°",
                    linestyle=styles[j], color=mpl_colors[i])
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    plt.ylabel(f"{field}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

def plot_avg_vel_prof(ds):
    
    print(f"Plotting angle-averaged velocity profile for {ds}.")
    
    # Create data object and get data
    ad = AMRData(ds, args.level, verbose=True)
    r, z = ad.position_data(units=False)
    v = ad["magvel"].d
    
    # Get 1d list of r values
    r1d = r[:,0]
    
    # Fixed resolution rays don't work in 2d with my yt version
    interp = RegularGridInterpolator((r1d, z[0]), v, bounds_error=False, fill_value=None)
    theta = np.linspace(0.0, np.pi, num=100)
    xi = np.column_stack((np.sin(theta), np.cos(theta)))

    avg = np.empty_like(r1d)
    for i in range(len(r1d)):
        pts = interp(r1d[i] * xi)
        avg[i] = pts.mean()
    
    plt.plot(r1d, avg)
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    plt.ylabel(r"$\bar{v}$ [cm/s]")
    plt.xscale("log")
    plt.yscale("log")

for ds in ts:
    
    if args.plot_prof:
        plot_prof(ds, ("magvel",), ("-",))
    if args.plot_avg_prof:
        plot_avg_vel_prof(ds)
        
    if not (args.plot_prof or args.plot_avg_prof):
        print("Nothing to do.")
        sys.exit(0)
        
    plt.savefig(f"vel_prof_{ds}.png")
    plt.gcf().clear()
