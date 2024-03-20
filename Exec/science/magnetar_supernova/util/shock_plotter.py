#!/usr/bin/env python3

import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-c', '--cmap')
parser.add_argument('--breakout_thresh', type=float, nargs='*', default=[1.1])
parser.add_argument('--window_size', type=int, nargs='*', default=[4])
parser.add_argument('--breakout_frac', type=float, nargs='*', default=[0.75])
parser.add_argument('--tmax_xs', type=float, nargs='*')
parser.add_argument('--tmax_tbo', type=float, nargs='*')
parser.add_argument('--tmax_fit', type=float, nargs='*')
parser.add_argument('--plot_int', type=int, nargs='*', default=[1])
parser.add_argument('--breakout_summary', action='store_true')
args = parser.parse_args()

class ArgList(list):
    
    def __init__(self, iterable):
        
        if iterable is None:
            super().__init__()
        else:
            super().__init__(iterable)
                
    def __getitem__(self, idx):
        
        if isinstance(idx, slice):
            outlist = []
            for i in range(slice.start, slice.stop, slice.step):
                outlist.append(super().__getitem__(i % len(self)))
            return outlist
        return super().__getitem__(idx % len(self))
        
args.breakout_thresh = ArgList(args.breakout_thresh)
args.window_size = ArgList(args.window_size)
args.breakout_frac = ArgList(args.breakout_frac)
args.tmax_xs = ArgList(args.tmax_xs)
args.tmax_tbo = ArgList(args.tmax_tbo)
args.tmax_fit = ArgList(args.tmax_fit)
args.plot_int = ArgList(args.plot_int)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if len(args.datafiles) < 1:
    sys.exit("No files were available to be loaded.")
    
def analytical_pos(ttild, A, alpha):
    
    return A * ttild**alpha
    
def analytical_vel(ttild, A, alpha):
    
    return A * alpha * ttild**(alpha - 1.)

for i, fname in enumerate(args.datafiles):
    
    f = h5py.File(fname, 'r')
    
    ############
    # Get data #
    ############
    
    t = f["time"][()]
    th = f["ang"][()]
    r_peak = f["shock_pos"]["r_peak"][()]
    r_outer = f["shock_pos"]["r_outer"][()]
    r_inner = f["shock_pos"]["r_inner"][()]
    
    dr, dz = f["dx"][()]
    dx = np.sqrt(dr**2 + dz**2)
    rmin, zmin = f["domain_left_edge"][()]
    rmax, zmax = f["domain_right_edge"][()]
    
    if "teng" in f.keys():
        teng, = f["teng"][()]
    else:
        teng = 2047.49866274
    R_t = 0.1 * teng * 0.02 * 2.997e10
    dttild = 0.01
    dxtild = dx / R_t
    if "label" in f.keys():
        label, = f["label"][()]
        label = label.decode('utf-8').strip(" '")
    else:
        label = str(i)
    Eeng = (2047.49866274 / teng) * 1.97392088e52
    Etild = 1e51 / (Eeng)
    
    ##################################################    
    # Replace spurious points in r_inner and r_outer #
    ##################################################
    
    mask = r_outer > (rmax - dx/2.)
    for j, row in enumerate(mask):
        if not row.any():
            continue
        r_outer[j, row] = np.interp(th[row], th[~row], r_outer[j, ~row])

    mask = r_inner <= (dx/2.)
    for j, row in enumerate(mask):
        if not row.any():
            continue
        r_inner[j, row] = np.interp(th[row], th[~row], r_inner[j, ~row])
        
    #########################################
    # Get dimensionless variables, colormap #
    #########################################
    
    ttild = t / teng
    r_peak_tild = r_peak / R_t
    
    if args.cmap:
        color = plt.cm.get_cmap(args.cmap)(i/(len(args.datafiles) - 1 + 1e-8))
    else:
        color = colors[i]
    
    #####################################    
    # Breakout time estimate with angle #
    #####################################
    
    if args.breakout_summary:
        
        defl_par = (r_inner.T / np.median(r_inner, axis=1)).T
        # defl_vel = np.empty_like(defl_par)
        # defl_vel[1:-1] = (defl_par[2:] - defl_par[:-2]) / (2 * dttild)
        # defl_vel[1] = defl_vel[-1] = 0.0
        defl_mask = (defl_par > args.breakout_thresh[i]).astype(np.float64)
        defl_mask = uniform_filter1d(defl_mask, size=args.window_size[i], axis=0, mode="constant",
                origin=-1)
        idx_bo = np.argmax(defl_mask >= args.breakout_frac[i], axis=0)
        idx_bo[idx_bo == 0] = -1
        shift_bo = 0.5 * (args.window_size[i] - np.ceil(args.window_size[i] * args.breakout_frac[i]))
        t_bo = t[idx_bo]
        ttild_bo = ttild[idx_bo]
            
    ##########################################
    # Power-law fit to median shell position #
    ##########################################
    
    if args.tmax_fit:
        if args.tmax_fit[i] < 0.0:
            tmax_fit = ttild_bo.min()
        else:
            tmax_fit = args.tmax_fit[i]
        mask = ttild < tmax_fit
        ttild_m = ttild[mask]
        r_peak_tild_m = r_peak_tild[mask]
    else:
        ttild_m = ttild
        r_peak_tild_m = r_peak_tild
        
    nth = r_peak_tild_m.shape[1]
    sigma = np.sqrt(np.pi * (2*nth + 1) / (4*nth)) * r_peak_tild_m.std(axis=1)
    fit_par, fit_cov = curve_fit(analytical_pos, ttild_m, np.median(r_peak_tild_m, axis=1),
            sigma=sigma)
    fit_std = np.sqrt(np.diag(fit_cov))
    A, alpha = fit_par
    
    #####################################  
    # Power-law fit to shell deflection #
    #####################################
    
    if args.tmax_tbo:
        mask = ttild < args.tmax_tbo[i]
        ttild_m = ttild[mask]
        r_peak_tild_m = r_peak_tild[mask]
    else:
        ttild_m = ttild
        r_peak_tild_m = r_peak_tild
        
    fit_delt_par, fit_delt_cov = curve_fit(analytical_pos, ttild[mask],
            r_peak_tild[mask].max(axis=1) - np.median(r_peak_tild[mask], axis=1))
    fit_delt_std = np.sqrt(np.diag(fit_delt_cov))
    Adelt, alphadelt = fit_delt_par
    t_first_bo = ((A * alpha) / (Adelt * alphadelt) * 0.6)**(1. / (alphadelt - alpha))
    
    ################## 
    # Console output #
    ##################
    
    print(f"{label}:")
    print(f"A = {fit_par[0]} ± {fit_std[0]}; α = {fit_par[1]} ± {fit_std[1]}")
    print(f"t_first_bo = {t_first_bo}")
    if args.breakout_summary:
        t_bo_idx = np.argsort(t_bo)
        for n_j, j in enumerate(t_bo_idx):
            eq_ang = np.abs(np.pi/2 - th[j])*180./np.pi
            print(f"({ttild_bo[j]:.2f}, {eq_ang:.1f})", end=" ")
            if (n_j+1) % 6 == 0:
                print()
    print('\n')
    
    #########
    # Plots #
    #########
    
    plt.figure(0)
    
    if args.tmax_xs:
        mask = ttild < args.tmax_xs[i]
        ttild_m = ttild[mask][::args.plot_int[i]]
        r_peak_tild_m = r_peak_tild[mask][::args.plot_int[i]]
    else:
        ttild_m = ttild[::plot_int]
        r_peak_tild_m = r_peak_tild[::plot_int]
        
    # yerr = np.sqrt(np.pi * (2*nth + 1) / (4*nth)) * dxtild / np.sqrt(nth)
    rp_med = np.median(r_peak_tild_m, axis=1)
    rp_min = r_peak_tild_m.min(axis=1)
    rp_max = r_peak_tild_m.max(axis=1)
    yerr = [rp_med - rp_min, rp_max - rp_med]
    plt.errorbar(ttild_m / Etild, rp_med / Etild, yerr=yerr/Etild, fmt=".", label=label,
            markersize=8, capsize=2, color=color)
    # plt.errorbar(ttild_m * Etild, r_peak_tild_m.min(axis=1), yerr=dxtild, fmt="^",
    #          markersize=4, capsize=2, color=color)
    # plt.errorbar(ttild_m * Etild, r_peak_tild_m.max(axis=1), yerr=dxtild, fmt="v",
    #          markersize=4, capsize=2, color=color)
    
    plt.plot(ttild_m / Etild, analytical_pos(ttild_m, *fit_par) / Etild, color=color, linestyle="--")
    
    plt.figure(1)
    plt.plot(ttild_m / Etild, (rp_max - rp_med) / Etild, color=color, label=label)
    plt.plot(ttild_m / Etild, analytical_pos(ttild_m, *fit_delt_par) / Etild, color=color, linestyle="--")
    
    f.close()

plt.figure(0)
plt.xlabel(r'$\tilde{t}~/~\tilde{E}_{\mathrm{kin,ej}}$')
plt.ylabel(r'$\tilde{x}_{s}~/~\tilde{E}_{\mathrm{kin,ej}}$')
plt.legend()
plt.gcf().savefig("shock_pos_vs_time.png")

plt.figure(1)
plt.xlabel(r'$\tilde{t}~/~\tilde{E}_{\mathrm{kin,ej}}$')
plt.ylabel(r'$\tilde{delta}~/~\tilde{E}_{\mathrm{kin,ej}}$')
plt.legend()
plt.gcf().savefig("shell_defl_vs_time.png")
