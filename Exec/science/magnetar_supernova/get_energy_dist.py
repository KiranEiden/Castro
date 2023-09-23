#!/usr/bin/env python3

import yt
import sys
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from yt.frontends.boxlib.data_structures import CastroDataset
    
thresh = 8e9
plot_mask = False
plot_M_of_v = True

ts = sys.argv[1:]
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: CastroDataset(file.rstrip('/'))
ts = map(tf, ts)
    
def calc_2d(ds):
    
    slc = ds.slice(2, np.pi)
    nr, nz, _ = ds.domain_dimensions
    rlo, zlo, _ = ds.domain_left_edge
    rhi, zhi, _ = ds.domain_right_edge
    frb = yt.FixedResolutionBuffer(slc, (rlo, rhi, zlo, zhi), (nr, nz))
    
    r = frb[('index', 'r')].d
    if plot_mask:
        z = frb[('index', 'z')].d
    dr = rhi.d / nr
    dz = zhi.d / nz
    vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
    
    rhoE = frb[('boxlib', 'rho_E')].d
    rhoe = frb[('boxlib', 'rho_e')].d
    vel = frb[('boxlib', 'magvel')].d
    
    of_mask = vel > thresh
    ej_mask = ~of_mask

    if plot_mask:
        plt.scatter(r[of_mask], z[of_mask], s=0.5)
        plt.xlabel('r [cm]')
        plt.ylabel('z [cm]')
        plt.xlim(rlo, rhi)
        plt.ylim(zlo, zhi)
        plt.savefig(f'mask_{ds.current_time.d:.0f}.png')
        plt.gcf().clear()

    if plot_M_of_v:
        plot = yt.ProfilePlot(ds, ('boxlib', 'magvel'), [('gas', 'mass')], weight_field=None)
        plot.save()

    Etot_of = (rhoE[of_mask] * vol[of_mask]).sum()
    etot_of = (rhoe[of_mask] * vol[of_mask]).sum()
    Etot_ej = (rhoE[ej_mask] * vol[ej_mask]).sum()
    etot_ej = (rhoe[ej_mask] * vol[ej_mask]).sum()
    
    return Etot_of, Etot_ej, etot_of, etot_ej

with open('energy_dist.dat', 'w') as datfile:
    
    print('t', 'E_of', 'E_ej', 'e_of', 'e_ej', file=datfile)
    print('s', 'erg', 'erg', 'erg', 'erg', file=datfile)

    for ds in ts:
        
        n = ds.dimensionality
        fstr = f"calc_{n}d(ds)"
        E_of, E_ej, e_of, e_ej = eval(fstr)
        
        print(ds.current_time.d, E_of, E_ej, e_of, e_ej, file=datfile)
