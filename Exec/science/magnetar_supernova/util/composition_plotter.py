#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import analysis_util as au

prop_cycle = plt.rcParams['axes.prop_cycle']
mpl_colors = prop_cycle.by_key()['color']

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('--plot_avg_prof', action='store_true')
parser.add_argument('--use_mpi', action='store_true')
args = parser.parse_args()

if args.use_mpi:
    MPI = au.mpi_importer()
is_main_proc = (not args.use_mpi) or (MPI.COMM_WORLD.Get_rank() == 0)

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

if is_main_proc:
    print("Will load the following files: {}\n".format(ts))

ts = au.FileLoader(ts, args.use_mpi)

def to_color(x, idx, small, logmin, logrange):
    
    x[x < small[idx]] = small[idx]
    return (np.log10(x) - logmin[idx]) / logrange[idx]
    
# def plot_prof(ds, fields, styles):
# 
#     print(f"Plotting radial profiles for {ds}.")
# 
#     rlo, zlo, plo = ds.domain_left_edge
#     rhi, zhi, phi = ds.domain_right_edge
#     zero = 0.0 * rlo
# 
#     for i, th in enumerate(np.linspace(0.0, np.pi, num=5)):
# 
#         ray = ds.ray((zero,)*3, (rhi*np.sin(th), rhi*np.cos(th), zero))
# 
#         r = np.sqrt(ray[('index', 'r')]**2 + ray[('index', 'z')]**2)
#         idx = np.argsort(r)
# 
#         r = r[idx]
# 
#         for j, field in enumerate(fields):
# 
#             data = ray[field][idx]
#             plt.plot(r, data, label=f"{field} (θ = {th*180/np.pi}°)",
#                     linestyle=styles[j], color=mpl_colors[i])
# 
#     plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.legend()
# 
#     plt.savefig(f'comp_prof_{ds}.png')
#     plt.gcf().clear()

for ds in ts:
    
    # Make data object and retrieve data
    ad = au.AMRData(ds, args.level)
    r, z = ad.position_data(units=False)
        
    X_H = ad['X(H1)'].d
    X_O = ad['X(O16)'].d
    X_Ni = ad['X(Ni56)'].d

    small = np.array([1e-6, 1e-6, 1e-6])
    large = np.array([1.0, 1.0, 1.0])

    logmin = np.log10(small)
    logmax = np.log10(large)
    logrange = logmax - logmin
     
    red = to_color(X_H, 0, small, logmin, logrange)
    grn = to_color(X_O, 1, small, logmin, logrange)
    blu = to_color(X_Ni, 2, small, logmin, logrange)

    rgb = np.stack((red, grn, blu), axis=2)
    plt.imshow(np.swapaxes(rgb, 0, 1), extent=[r[0,0], r[-1,0], z[0,0], z[0,-1]])
    plt.xlabel("r [cm]")
    plt.ylabel("z [cm]")
    plt.savefig(f"composition_{ds}.png")
    plt.gcf().clear()
    
    if args.plot_avg_prof:
        
        H_prof = au.get_avg_prof_2d(ds, 100, r, z, X_H)
        O_prof = au.get_avg_prof_2d(ds, 100, r, z, X_O)
        Ni_prof = au.get_avg_prof_2d(ds, 100, r, z, X_Ni)
        
        plt.plot(r[:, 0], H_prof, label=r"$^{1}\mathrm{H}$")
        plt.plot(r[:, 0], O_prof, label=r"$^{16}\mathrm{O}$")
        plt.plot(r[:, 0], Ni_prof, label=r"$^{56}\mathrm{Ni}$")
        
        plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
        plt.ylabel(r"X")
        plt.yscale("log")
        plt.legend()
        
        plt.savefig(f'avg_comp_prof_{ds}.png')
        plt.gcf().clear()
