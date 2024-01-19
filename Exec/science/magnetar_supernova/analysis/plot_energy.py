#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

t, E, e, M = np.loadtxt('energy.dat', skiprows=2).T

plt.plot(t, E, label=r"$E$")
plt.plot(t, E-e, label=r"$\frac{1}{2}M|u|^2$")
plt.plot(t, e, label=r"$e$")
plt.plot(t, np.ones_like(t)*1.97e52, color='black', linestyle="--", label=r"$E_{0,mag}$")

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

plt.legend()

plt.savefig("energy_vs_time.png")
