#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
files = sorted(glob.glob("energy_dist_*.dat"))

for i, f in enumerate(files):

    P_0 = float(f.split("_")[2].split(".")[0][:-2])
    t, E_of, E_ej, e_of, e_ej = np.loadtxt(f, skiprows=2).T
    
    t_m = 2047.49866274 * P_0**2
    E_0 = 1.97392088e+52 / P_0**2
    frac_emit = 1. - 1. / (t/t_m + 1.)
    E_emit = frac_emit*E_0
    E_tot = E_emit/2. + E_ej[0]

    plt.scatter(t/t_m, E_ej/(E_of + E_ej), label=r"$E_{\mathrm{ej}}$" + f" ({P_0} ms)", marker=".", color=colors[i])
    plt.scatter(t/t_m, E_of/(E_of + E_ej), label=r"$E_{\mathrm{of}}$" + f" ({P_0} ms)", marker="^", color=colors[i])
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
