#!/usr/bin/env python3

import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument("--P_0", nargs=1, type=float)
parser.add_argument("--t_ramp", nargs=1, type=float)
parser.add_argument("--E_sn", nargs=1)
args = parser.parse_args()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if not args.datafiles:
    args.datafiles = sorted(glob.glob("energy_*.dat"))
assert args.datafiles, "No energy data files available to plot."

if not args.P_0:
    args.P_0 = [float(f.split("_")[1].split(".")[0][:-2]) for f in files]
elif len(args.P_0) == 1:
    args.P_0 = args.P_0 * len(args.datafiles)
else:
    assert len(args.P_0) == len(args.datafiles)
    
if not args.t_ramp:
    args.t_ramp = [0.05] * len(args.datafiles)
elif len(args.t_ramp) == 1:
    args.t_ramp = args.t_ramp * len(args.datafiles)
else:
    assert len(args.t_ramp) == len(args.datafiles)

for i, f in enumerate(args.datafiles):

    t, E_of, E_ej, e_of, e_ej, _, _ = np.loadtxt(f, skiprows=2).T
    P_0 = args.P_0[i]
    t_ramp = args.t_ramp[i]
    
    if args.E_sn is None:
        args.E_sn = 1e51
    elif args.E_sn == 'init_E':
        args.E_sn = E_ej[0]
    else:
        args.E_sn = float(args.E_sn)
    
    t_m = 2047.49866274 * P_0**2
    E_0 = 1.97392088e+52 / P_0**2
    frac_emit = np.log(t_ramp + 1.) / t_ramp - 1. / (t/t_m + 1.)
    E_emit = frac_emit*E_0
    E_tot = E_emit + args.E_sn
    
    E_rat = (E_ej + E_of) / E_tot
    delt = np.abs(E_rat - 1.)
    if np.any(delt > 1e-3):
        if isinstance(delt, float):
            print("Total energy differs from theoretical value by more than 0.1%." +
                    f" Ratio E_sim/E_theory is {E_rat}.")
        else:
            i = np.argmax(delt)
            print("Total energy differs from theoretical value by more than 0.1%." +
                    f" Maximum difference is {delt[i]} at t = {t[i]}.")

    plt.scatter(t/t_m, E_ej/E_tot, label=r"$E_{\mathrm{ej}}$" + f" ({P_0} ms)", marker=".", color=colors[i])
    plt.scatter(t/t_m, 1. - E_ej/E_tot, label=r"$E_{\mathrm{of}}$" + f" ({P_0} ms)", marker="^", color=colors[i])
    #plt.plot(t, np.ones_like(t)*1.97e52, color='black', linestyle="--", label=r"$E_{0,mag}$")

    #t_m = 2047.49866274
    #E_0 = 1.97392088e+52
    #frac_emit = 1. - 1. / (2*t/t_m + 1.)
    #E_emit = E_0 * frac_emit

    # x, _, _, _ = np.linalg.lstsq(E_emit.reshape(len(E_emit), 1), (E - E[0]).reshape(len(E), 1), rcond=None)
    # x = x[0,0]

    # plt.plot(t, E)
    # plt.plot(t, x*E_emit + E[0])

plt.xlabel(r"$t/t_{\mathrm{mag}}$")
plt.ylabel("Energy Fraction")

plt.legend()

plt.savefig("energy_dist_vs_time.png")
