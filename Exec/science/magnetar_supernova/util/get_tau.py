#!/usr/bin/env python3

import yt
import sys
import numpy as np
import unyt as u

from scipy.integrate import simpson
from yt.frontends.boxlib.data_structures import CastroDataset

ts = sys.argv[1:]
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: CastroDataset(file.rstrip('/'))
ts = map(tf, ts)

for ds in ts:
    
    with open(f'tau_{ds}.dat', 'w') as datfile:

        print('theta', 'N', 'tau', file=datfile)
        print('rad', 'cm**-2', 'dimensionless', file=datfile)
        
        rlo, zlo, plo = ds.domain_left_edge
        rhi, zhi, phi = ds.domain_right_edge
        
        for th in np.linspace(0.0, np.pi/2, num=500):
            
            dist = max(rhi - rlo, zhi - zlo)
            ray = ds.ray((rlo, zlo, plo), (dist*np.cos(th)+rlo, dist*np.sin(th)+zlo, plo))
            
            r = np.sqrt(ray[('gas', 'r')]**2 + ray[('gas', 'z')]**2)
            idx = np.argsort(r)
            
            r = r[idx]
            rho = ray['density'][idx]
            
            cs_thom = 6.65e-24 * u.cm**2
            amu = 1.660539e-24 * u.g
            
            col_dens = simpson((rho / amu).d, r.d) * u.cm**-2
            tau = col_dens * cs_thom
            
            print(th, col_dens.d, tau.d, file=datfile)
