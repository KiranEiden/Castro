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
parser.add_argument('-t0', '--time_offset', type=float)
parser.add_argument('--no_rgb', action='store_true')
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

for ds in ts:
    
    # Make data object and retrieve data
    ad = au.AMRData(ds, args.level)
    r, z = ad.position_data(units=False)
    dr, dz = ad.dds[:, args.level].d
    
    X_H = ad['X(H1)'].d
    X_O = ad['X(O16)'].d
    X_Ni = ad['X(Ni56)'].d
    X_He = ad['X(He4)'].d

    if args.plot_avg_prof is not None:
        
        vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
        cell_mass = ad['density'].d * vol
    
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
        X_He = make_slc(X_He)
        cell_mass = make_slc(cell_mass)
        
    small = np.array([1e-6, 1e-6, 1e-6])
    large = np.array([1.0, 1.0, 1.0])

    logmin = np.log10(small)
    logmax = np.log10(large)
    logrange = logmax - logmin
        
    if args.plot_avg_prof is not None:
        
        if args.plot_avg_prof:
            opt = get_argdict(args.plot_avg_prof)
        else:
            opt = dict()
        
        H_prof = au.get_avg_prof_2d(ds, 100, r, z, X_H, weight_data=cell_mass)
        O_prof = au.get_avg_prof_2d(ds, 100, r, z, X_O, weight_data=cell_mass)
        Ni_prof = au.get_avg_prof_2d(ds, 100, r, z, X_Ni, weight_data=cell_mass)
        He_prof = au.get_avg_prof_2d(ds, 100, r, z, X_He, weight_data=cell_mass)

        if args.time_offset is not None:
            x = r[:, 0] / (ds.current_time.d + args.time_offset)
            xlabel = r"$R/t$ [cm/s]"
        else:
            x = r[:, 0]
            xlabel = r"$R$ [cm]"
        
        plt.plot(x, H_prof, label=r"$^{1}\mathrm{H}$")
        plt.plot(x, O_prof, label=r"$^{16}\mathrm{O}$")
        plt.plot(x, Ni_prof, label=r"$^{56}\mathrm{Ni}$")
        plt.plot(x, He_prof, label=r"$^{4}\mathrm{He}$")
        
        plt.xlabel(xlabel)
        plt.ylabel(r"X")
        if opt.get("xlog", False):
            plt.xscale("log")
        plt.yscale("log")
        plt.ylim(small.min(), 1.2)
        plt.legend()
        
        plt.savefig(f'avg_comp_prof_{ds}.png')
        plt.gcf().clear()
     
    if not args.no_rgb:

        red = to_color(X_H, 0, small, logmin, logrange)
        grn = to_color(X_O, 1, small, logmin, logrange)
        blu = to_color(X_Ni, 2, small, logmin, logrange)

        rgb = np.stack((red, grn, blu), axis=2)
        plt.imshow(np.swapaxes(rgb, 0, 1), extent=[r[0,0], r[-1,0], z[0,0], z[0,-1]])
        plt.xlabel("r [cm]")
        plt.ylabel("z [cm]")
        plt.savefig(f"composition_{ds}.png")
        plt.gcf().clear()
