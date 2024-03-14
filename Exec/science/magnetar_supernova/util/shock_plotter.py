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
parser.add_argument('--breakout_thresh', type=float, default=1.1)
parser.add_argument('--window_size', type=int, default=4)
parser.add_argument('--breakout_frac', type=float, default=0.75)
parser.add_argument('--tmax_xs', type=float)
parser.add_argument('--tmax_tbo', type=float)
parser.add_argument('--tmax_fit', type=float)
args = parser.parse_args()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if len(args.datafiles) < 1:
    sys.exit("No files were available to be loaded.")
    
def analytical_func(ttild, A, alpha):
    
    return A * ttild**alpha

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
    
    if "teng" in f.keys():
        teng, = f["teng"][()]
    else:
        teng = 2047.49866274
    R_t = 0.1 * teng * 0.02 * 2.997e10
    dttild = 0.01
    drtild = 7.5e13/16384 / R_t
    if "label" in f.keys():
        label, = f["label"][()]
        label = label.decode('utf-8').strip(" '")
    else:
        label = str(i)
    
    ##################################################    
    # Replace spurious points in r_inner and r_outer #
    ##################################################
    
    mask = r_outer > 7.4e13
    for j, row in enumerate(mask):
        if not row.any():
            continue
        r_outer[j, row] = np.interp(th[row], th[~row], r_outer[j, ~row])

    mask = r_inner <= (7.5e13/16384*2. * 1.01)
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
    
    #############################    
    # Breakout time measurement #
    #############################
    
    defl_par = (r_inner.T / np.median(r_inner, axis=1)).T
    # defl_vel = np.empty_like(defl_par)
    # defl_vel[5:] = (defl_par[5:] - defl_par[:-5]) / (5 * dttild)
    # defl_vel[:5] = 0.0
    defl_mask = (defl_par > args.breakout_thresh).astype(np.float64)
    defl_mask = uniform_filter1d(defl_mask, size=args.window_size, axis=0, mode="constant", origin=-1)
    idx_bo = np.argmax(defl_mask >= args.breakout_frac, axis=0)
    idx_bo[idx_bo == 0] = -1
    shift_bo = 0.5 * (args.window_size - np.ceil(args.window_size * args.breakout_frac))
    t_bo = t[idx_bo]
    ttild_bo = ttild[idx_bo]
            
    #################  
    # Power-law fit #
    #################
    
    if args.tmax_fit:
        if args.tmax_fit < 0.0:
            args.tmax_fit = ttild_bo.min()
        mask = ttild < args.tmax_fit
        ttild_m = ttild[mask]
        r_peak_tild_m = r_peak_tild[mask]
    else:
        ttild_m = ttild
        r_peak_tild_m = r_peak_tild
        
    nth = r_peak_tild_m.shape[1]
    sigma = np.sqrt(np.pi * (2*nth + 1) / (4*nth)) * r_peak_tild_m.std(axis=1)
    fit_par, fit_cov = curve_fit(analytical_func, ttild_m, np.median(r_peak_tild_m, axis=1),
            sigma=sigma)
    fit_std = np.sqrt(np.diag(fit_cov))
    fit_data = analytical_func(ttild, *fit_par)
    
    ################## 
    # Console output #
    ##################
    
    print(f"{label}:")
    print(f"A = {fit_par[0]} ± {fit_std[0]}; α = {fit_par[1]} ± {fit_std[1]}")
    t_bo_idx = np.argsort(t_bo)
    for n_j, j in enumerate(t_bo_idx):
        eq_ang = np.abs(np.pi/2 - th[j])*180./np.pi
        print(f"({1.3*ttild_bo[j]:.2f}, {eq_ang:.1f})", end=" ")
        if (n_j+1) % 6 == 0:
            print()
    print()
    
    #########
    # Plots #
    #########
    
    plt.figure(0)
    if args.tmax_tbo:
        mask = ttild_bo < args.tmax_tbo
        plt.plot(th[mask], ttild_bo[mask], label=label, color=color)
    else:
        plt.plot(th, ttild_bo, label=label, color=color)
    
    if args.tmax_xs:
        mask = ttild < args.tmax_xs
        ttild_m = ttild[mask]
        r_peak_tild_m = r_peak_tild[mask]
        fit_data_m = fit_data[mask]
    else:
        ttild_m = ttild
        r_peak_tild_m = r_peak_tild
        fit_data_m = fit_data
    
    plt.figure(1)
    # yerr = np.sqrt(np.pi * (2*nth + 1) / (4*nth)) * drtild / np.sqrt(nth)
    rp_med = np.median(r_peak_tild_m, axis=1)
    rp_min = r_peak_tild_m.min(axis=1)
    rp_max = r_peak_tild_m.max(axis=1)
    yerr = [rp_med - rp_min, rp_max - rp_med]
    plt.errorbar(ttild_m, rp_med, yerr=yerr, fmt=".", label=label,
            markersize=8, capsize=2, color=color)
    # plt.errorbar(ttild_m, r_peak_tild_m.min(axis=1), yerr=drtild, fmt="^",
    #          markersize=4, capsize=2, color=color)
    # plt.errorbar(ttild_m, r_peak_tild_m.max(axis=1), yerr=drtild, fmt="v",
    #          markersize=4, capsize=2, color=color)
    
    plt.plot(ttild_m, fit_data_m, color=color, linestyle="--")
            
    #plt.errorbar(t, r_inner.mean(axis=1), yerr=r_peak.std(axis=1), xerr=dt, fmt="^",
    #        markersize=8, capsize=2, color=color)
    #plt.errorbar(t, r_outer.mean(axis=1), yerr=r_peak.std(axis=1), xerr=dt, fmt="v",
    #        markersize=8, capsize=2, color=color)
    
    f.close()

plt.figure(0)
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$\tilde{t}_{\mathrm{bo}}$')
plt.legend()
plt.gcf().savefig("t_bo_vs_ang.png")

plt.figure(1)
plt.xlabel(r'$\tilde{t}$')
plt.ylabel(r'$\tilde{x}_{s}$')
plt.legend()
plt.gcf().savefig("shock_pos_vs_time.png")
