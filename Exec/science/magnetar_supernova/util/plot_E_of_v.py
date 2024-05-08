#!/usr/bin/env python3

import yt
import argparse
import unyt as u
import numpy as np
import matplotlib.pyplot as plt
import analysis_util as au

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs='+')
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('-n', '--nbins', type=int, default=100)
parser.add_argument('-c', '--cmap')
parser.add_argument('-C', '--colors', nargs='+')
parser.add_argument('-L', '--labels', nargs='+')
parser.add_argument('-o', '--out', default='E_of_v.png')
parser.add_argument('--logx', action='store_true')
parser.add_argument('--logy', action='store_true')
parser.add_argument('--shell_rad', nargs='+')
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

ts = (yt.load(ds, hint='CastroDataset') for ds in ts)

if args.shell_rad:
    assert len(args.shell_rad) == len(args.datafiles)
    args.shell_rad = list(map(float, args.shell_rad))

if not args.colors:
    prop_cycle = plt.rcParams['axes.prop_cycle']
    args.colors = prop_cycle.by_key()['color']
    
for i, ds in enumerate(ts):
    
    if args.cmap:
        color = plt.cm.get_cmap(args.cmap)(i/(len(args.datafiles) - 1 + 1e-8))
    else:
        color = args.colors[i]
        
    if args.labels:
        label = args.labels[i]
    else:
        label = f"t = {ds.current_time}"
    
    ad = au.AMRData(ds, args.level)
    
    r, z = ad.position_data(units=False)
    dr, dz = ad.dds[:, args.level].d
    vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
    
    v = ad['magvel'].d
    rhoE = ad['rho_E'].d
    rhoe = ad['rho_e'].d
    
    vmin = v.min() * (1.0 - 1e-5)
    vmax = v.max() * (1.0 + 1e-5)
    bins = np.linspace(vmin, vmax, num=(args.nbins+1))
    Etot = np.zeros(len(bins)-1)
    etot = np.zeros(len(bins)-1)
    
    if args.shell_rad:
        # Mask out central cavity
        ej_mask = r**2 + z**2 >= args.shell_rad[i]**2
    
    for j in range(len(bins)-1):
        
        mask = np.logical_and(bins[j] <= v, v < bins[j+1])
        if args.shell_rad:
            mask &= ej_mask
        Etot[j] += (rhoE[mask] * vol[mask]).sum()
        etot[j] += (rhoe[mask] * vol[mask]).sum()
    
    beta_samp = (bins[:-1] + bins[1:]) / 2.0 / u.c.in_cgs().d
    dbeta = (bins[1] - bins[0]) / u.c.in_cgs().d
    plt.plot(beta_samp, Etot / dbeta, color=color, label=label)
    # plt.plot(beta_samp, etot / dbeta, color=color, linestyle=':')
    # plt.plot(beta_samp, (Etot - etot) / dbeta, color=color, linestyle='--')
    
plt.xlabel(r"$\beta = v/c$")
plt.ylabel(r"$E/ \Delta \beta$ (erg)")
if args.logx:
    plt.xscale("log")
if args.logy:
    plt.yscale("log")
plt.legend()
plt.gcf().savefig(args.out)
