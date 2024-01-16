#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from analysis_util import AMRData

prop_cycle = plt.rcParams['axes.prop_cycle']
mpl_colors = prop_cycle.by_key()['color']

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('--plot_prof', action='store_true')
parser.add_argument('--plot_avg_prof', action='store_true')
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
ts = map(tf, ts)

def to_color(x, idx):
    
    x[x < small[idx]] = small[idx]
    return (np.log10(x) - logmin[idx]) / logrange[idx]
    
def plot_prof(ds, fields, styles):
    
    print(f"Plotting radial profiles for {ds}.")
    
    rlo, zlo, plo = ds.domain_left_edge
    rhi, zhi, phi = ds.domain_right_edge
    zero = 0.0 * rlo
    
    for i, th in enumerate(np.linspace(0.0, np.pi, num=5)):
        
        ray = ds.ray((zero,)*3, (rhi*np.sin(th), rhi*np.cos(th), zero))
    
        r = np.sqrt(ray[('index', 'r')]**2 + ray[('index', 'z')]**2)
        idx = np.argsort(r)
    
        r = r[idx]
    
        for j, field in enumerate(fields):
        
            data = ray[field][idx]
            plt.plot(r, data, label=f"{field} (θ = {th*180/np.pi}°)",
                    linestyle=styles[j], color=mpl_colors[i])
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    
    plt.savefig(f'comp_prof_{ds}.png')
    plt.gcf().clear()

def plot_avg_prof(ds, r, z, Xs, labels):
    
    print(f"Plotting angle-averaged radial mass fraction profiles for {ds}.")
    
    # Get 1d list of r values
    r1d = r[:,0]
    
    # Fixed resolution rays don't work in 2d with my yt version
    for i, X in enumerate(Xs):
        
        # Fixed resolution rays don't work in 2d with my yt version
        interp = RegularGridInterpolator((r1d, z[0]), X, bounds_error=False, fill_value=None)
        theta = np.linspace(0.0, np.pi, num=100)
        xi = np.column_stack((np.sin(theta), np.cos(theta)))
        
        avg = np.empty_like(r1d)
        for j in range(len(r1d)):
            pts = interp(r1d[j] * xi)
            avg[j] = pts.mean()
            
        plt.plot(r1d, avg, label=labels[i])
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    plt.ylabel(r"X")
    plt.yscale("log")
    plt.legend()
    
    plt.savefig(f'avg_comp_prof_{ds}.png')
    plt.gcf().clear()

for ds in ts:
    
    # Make data object and retrieve data
    ad = AMRData(ds, args.level, verbose=True)
        
    X_H = ad['X(H1)'].d
    X_O = ad['X(O16)'].d
    X_Ni = ad['X(Ni56)'].d

    small = np.array([1e-10, 1e-10, 1e-10])
    large = np.array([1.0, 1.0, 1.0])

    logmin = np.log10(small)
    logmax = np.log10(large)
    logrange = logmax - logmin
     
    red = to_color(X_H, 0)
    grn = to_color(X_O, 1)
    blu = to_color(X_Ni, 2)

    rgb = np.stack((red, grn, blu), axis=2)
    plt.imshow(np.swapaxes(rgb, 0, 1))
    plt.savefig(f"composition_{ds}.png")
    plt.gcf().clear()
    
    if args.plot_prof:
        plot_prof(ds, ("X(H1)", "X(O16)", "X(Ni56)"), ("-", "--", ":"))
    
    if args.plot_avg_prof:
        r, z = ad.position_data(units=False)
        plot_avg_prof(ds, r, z, (X_H, X_O, X_Ni),
                (r"$^{1}\mathrm{H}$", r"$^{16}\mathrm{O}$", r"$^{56}\mathrm{Ni}$"))
