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
        
        rho_sinth = ad['density'].d * r / np.sqrt(r**2 + z**2)
    
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
        if args.plot_avg_prof is not None:
            rho_sinth = make_slc(rho_sinth)
        
    small = np.array([1e-5, 1e-5, 1e-5])
    large = np.array([1.0, 1.0, 1.0])

    logmin = np.log10(small)
    logmax = np.log10(large)
    logrange = logmax - logmin
    
    plt.rc('axes', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=14)
        
    if args.plot_avg_prof is not None:
        
        if args.plot_avg_prof:
            opt = get_argdict(args.plot_avg_prof)
        else:
            opt = dict()
       
        H_prof = au.get_avg_prof_2d(ds, 100, r, z, X_H, weight_data=rho_sinth)
        O_prof = au.get_avg_prof_2d(ds, 100, r, z, X_O, weight_data=rho_sinth)
        Ni_prof = au.get_avg_prof_2d(ds, 100, r, z, X_Ni, weight_data=rho_sinth)
        He_prof = au.get_avg_prof_2d(ds, 100, r, z, X_He, weight_data=rho_sinth)

        if args.time_offset is not None:
            x = r[:, 0] / (ds.current_time.d + args.time_offset)
            xlabel = r"$R/(t + t_0)$ (cm/s)"
        else:
            x = r[:, 0]
            xlabel = r"$R$ (cm)"
        
        plt.plot(x, H_prof, label=r"$X_{\mathrm{env}}$")
        plt.plot(x, O_prof, label=r"$X_{\mathrm{ims}}$")
        plt.plot(x, Ni_prof, label=r"$X_{\mathrm{core}}$")
        plt.plot(x, He_prof, label=r"$X_{\mathrm{wind}}$")
        
        plt.xlabel(xlabel)
        plt.ylabel(r"Average Mass Fraction")
        if opt.get("xlog", False):
            plt.xscale("log")
        if "xmax" in opt:
            plt.xlim(x.min(), float(opt['xmax']))
        plt.yscale("log")
        plt.ylim(small.min(), 1.2)
        if ds.current_time.d < 1e4:
            plt.legend(loc="lower right")
        else:
            plt.legend(loc="lower center")
        
        plt.savefig(f'avg_comp_prof_{ds}.pdf', dpi=480, bbox_inches='tight')
        plt.gcf().clear()
     
    if not args.no_rgb:

        red = to_color(X_He, 0, small, logmin, logrange)
        grn = to_color(X_O, 1, small, logmin, logrange)
        blu = to_color(X_Ni, 2, small, logmin, logrange)

        rgb = np.stack((red, grn, blu), axis=2)
        plt.imshow(np.swapaxes(rgb, 0, 1), extent=[r[0,0], r[-1,0], z[0,0], z[0,-1]])
        plt.xlabel("r (cm)")
        plt.ylabel("z (cm)")
        plt.gcf().set_size_inches((6.85, 9.2))
        xpos = 1.0 - (0.9 * (1.0 - (5.5e11/1.2e12)))
        plt.text(xpos*7.5e13, (1.075e12/1.2e12)*7.5e13, r"$5.00~\mathrm{t_{eng}}$ (2.84 hr)", color='white', fontsize="large")
        plt.savefig(f"composition_{ds}.pdf", dpi=480)
        plt.gcf().clear()
