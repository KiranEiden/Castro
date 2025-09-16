#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

dpi = 480

if len(sys.argv) > 1:
    fnames = sys.argv[1:]
    fiter = map(h5py.File, fnames)
else:
    print("Usage: ./plot_Trad.py <plt_files>")
    
plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

for fname, f in zip(fnames, fiter):
    
    fname_base = '.'.join(fname.split('.')[:-1])
    print(fname_base)
    
    T_rad = f["T_rad"][()]
    r = f["r"][()]
    z = f["z"][()]
    time = f["time"][()][0] / 86400
    
    log_T = np.log10(T_rad)
    
    axim = plt.imshow(np.swapaxes(log_T, 0, 1), cmap="afmhot",
            extent=[r.min(), r.max(), z.min(), z.max()], vmin=2.0, vmax=5.5)
    cbar = plt.colorbar(axim, fraction=0.075, pad=0.0024, aspect=len(z)/len(r)*20)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$\log[T_{\mathrm{rad}}~(\mathrm{K})]$", size=16)
    plt.xlabel("r (cm)", loc="center")
    plt.ylabel("z (cm)", loc="center")
    plt.annotate(f"{time:.1f} d", (r.min()*5.0, z.max()*0.9375), color="skyblue",
            fontsize="large")
    plt.tight_layout()
    plt.gcf().set_size_inches((6,9))
    fw, fh = dpi * plt.gcf().get_size_inches()
    fw = 2 * round(fw / 2)
    fh = 2 * round(fh / 2)
    if fw > (fw/dpi * dpi):
        fw += fw - (fw/dpi * dpi)
    if fh > (fh/dpi * dpi):
        fh += fh - (fh/dpi * dpi)
    plt.gcf().set_size_inches(fw/dpi, fh/dpi)
    plt.savefig(f"{fname_base}_log_T_rad.pdf", dpi=dpi)
    plt.gcf().clear()
    
    axim = plt.imshow(np.swapaxes(T_rad, 0, 1), cmap="afmhot",
            extent=[r.min(), r.max(), z.min(), z.max()], vmin=1e2, vmax=8e4)
    cbar = plt.colorbar(axim, fraction=0.15, pad=0.002, aspect=len(z)/len(r)*20)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$T_{\mathrm{rad}}~(\mathrm{K})$", size=16)
    plt.xlabel("r (cm)", loc="center")
    plt.ylabel("z (cm)", loc="center")
    plt.annotate(f"{time:.1f} d", (r.min()*6.0, z.max()*0.93), color="skyblue",
            fontsize="large")
    plt.tight_layout()
    plt.gcf().set_size_inches((6,9))
    fw, fh = dpi * plt.gcf().get_size_inches()
    fw = 2 * round(fw / 2)
    fh = 2 * round(fh / 2)
    if fw > (fw/dpi * dpi):
        fw += fw - (fw/dpi * dpi)
    if fh > (fh/dpi * dpi):
        fh += fh - (fh/dpi * dpi)
    plt.gcf().set_size_inches(fw/dpi, fh/dpi)
    plt.savefig(f"{fname_base}_T_rad.pdf", dpi=dpi)
    plt.gcf().clear()
