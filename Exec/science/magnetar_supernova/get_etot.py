#!/usr/bin/env python3

import yt
import sys
import numpy as np
import unyt as u
from yt.frontends.boxlib.data_structures import CastroDataset
    
ts = sys.argv[1:]
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: CastroDataset(file.rstrip('/'))
ts = map(tf, ts)

def calc_1d(ds):
    
    ray = ds.ortho_ray(0, (0, 0))
    idx = np.argsort(ray[('gas', 'r')].d)
    r = ray[('gas', 'r')].d[idx]
    dr = r[1] - r[0]
    vol = 4.*np.pi/3. * ((r+dr)**3 - r**3)
    
    rhoE = ray['rho_E'].d[idx]
    rhoe = ray['rho_e'].d[idx]
    rho = ray[('gas', 'density')].d[idx]
    
    Etot = (rhoE * vol).sum()
    etot = (rhoe * vol).sum()
    Mtot = (rho * vol).sum()
    
    return Etot, etot, Mtot
    
def calc_2d(ds):
    
    slc = ds.slice(2, np.pi)
    nr, nz, _ = ds.domain_dimensions
    rlo, zlo, _ = ds.domain_left_edge
    rhi, zhi, _ = ds.domain_right_edge
    frb = yt.FixedResolutionBuffer(slc, (rlo, rhi, zlo, zhi), (nr, nz))
    
    r = frb[('index', 'r')].d
    dr = rhi / nr
    dz = zhi / nz
    vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
    
    rhoE = frb[('boxlib', 'rho_E')].d
    rhoe = frb[('boxlib', 'rho_e')].d
    rho = frb[('gas', 'density')].d
    
    Etot = (rhoE * vol).sum()
    etot = (rhoe * vol).sum()
    Mtot = (rho * vol).sum()
    
    return Etot, etot, Mtot

with open('energy.dat', 'w') as datfile:
    
    print('t', 'E', 'e', 'M', file=datfile)
    print('s', 'erg', 'erg', 'g', file=datfile)

    for ds in ts:
        
        n = ds.dimensionality
        fstr = f"calc_{n}d(ds)"
        Etot, etot, Mtot = eval(fstr)
        
        print(ds.current_time.d, Etot, etot, Mtot, file=datfile)