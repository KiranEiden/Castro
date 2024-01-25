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
parser.add_argument('--plot_avg_prof', nargs='*', default=None)
parser.add_argument('-x', '--xlim', nargs=2, type=float)
parser.add_argument('-y', '--ylim', nargs=2, type=float)
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

def safe_convert(string_val):
    """
    Convert string value to an int or float if possible, and return it if not.
    Can also convert comma-separated lists of these types.
    """

    values = string_val.split(',')

    for i in range(len(values)):

        try:
            values[i] = int(values[i])
        except ValueError:
            pass

        try:
            values[i] = float(values[i])
        except ValueError:
            pass

    if len(values) == 1:
        values = values[0]
    return values

def get_argdict(arglist):
    """
    Converts whitespace-delimited list of key-value pairs into dictionary. Can handle numeric
    and list values.
    """

    pairs = map(lambda s: s.split(':'), arglist)
    pairs = ((k, safe_convert(v)) for k, v in pairs)
    return dict(pairs)

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
    
    if args.xlim or args.ylim:
        if not args.xlim:
            args.xlim = ad.left_edge[0], ad.right_edge[0]
        if not args.ylim:
            args.ylim = ad.left_edge[1], ad.right_edge[1]
            
        idx, _ = ad.region_idx(*args.xlim, *args.ylim)
        make_slc = lambda arr: arr[slice(*idx[0]), slice(*idx[1])]
        
        r = make_slc(r)
        z = make_slc(z)
        X_H = make_slc(X_H)
        X_O = make_slc(X_O)
        X_Ni = make_slc(X_Ni)

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
    
    if args.plot_avg_prof is not None:
        
        if args.plot_avg_prof:
            opt = get_argdict(args.plot_avg_prof)
        else:
            opt = dict()
        
        H_prof = au.get_avg_prof_2d(ds, 100, r, z, X_H)
        O_prof = au.get_avg_prof_2d(ds, 100, r, z, X_O)
        Ni_prof = au.get_avg_prof_2d(ds, 100, r, z, X_Ni)
        
        plt.plot(r[:, 0], H_prof, label=r"$^{1}\mathrm{H}$")
        plt.plot(r[:, 0], O_prof, label=r"$^{16}\mathrm{O}$")
        plt.plot(r[:, 0], Ni_prof, label=r"$^{56}\mathrm{Ni}$")
        
        plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
        plt.ylabel(r"X")
        if opt.get("xlog", False):
            plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        
        plt.savefig(f'avg_comp_prof_{ds}.png')
        plt.gcf().clear()
