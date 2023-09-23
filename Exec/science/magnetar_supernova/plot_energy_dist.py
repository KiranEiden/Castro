#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

t, E_of, E_ej, e_of, e_ej = np.loadtxt('energy_dist.dat', skiprows=2).T

plt.scatter(t, E_of, label=r"$E_{\mathrm{of}}$")
plt.scatter(t, E_ej, label=r"$E_{\mathrm{ej}}$")
#plt.plot(t, np.ones_like(t)*1.97e52, color='black', linestyle="--", label=r"$E_{0,mag}$")

# t_m = 2047.49866274
# E_0 = 1.97392088e+52
# frac_emit = 1. - 1. / (2*t/t_m + 1.)
# E_emit = E_0 * frac_emit

# x, _, _, _ = np.linalg.lstsq(E_emit.reshape(len(E_emit), 1), (E - E[0]).reshape(len(E), 1), rcond=None)
# x = x[0,0]

# plt.plot(t, E)
# plt.plot(t, x*E_emit + E[0])

plt.xlabel("t [s]")
plt.ylabel("Energy [erg]")

plt.yscale("log")

plt.legend()

plt.savefig("energy_dist_vs_time.png")
