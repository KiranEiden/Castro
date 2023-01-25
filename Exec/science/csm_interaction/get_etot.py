#!/usr/bin/env python3

import yt
import sys
import numpy as np
import matplotlib.pyplot as plt
from yt.frontends.boxlib.data_structures import CastroDataset
    
ts = sys.argv[1:]
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: CastroDataset(file.rstrip('/'))
ts = map(tf, ts)

with open('energy.dat', 'w') as datfile:
    
    print('t', 'E', 'e', 'M', file=datfile)
    print('s', 'erg', 'erg', 'g', file=datfile)

    for ds in ts:
        
        ray = ds.ortho_ray(0, (0, 0))
        idx = np.argsort(ray[('gas', 'r')].d)
        r = ray[('gas', 'r')].d[idx]
        dr = r[1] - r[0]
        rhoE = ray['rho_E'].d[idx]
        rhoe = ray['rho_e'].d[idx]
        rho = ray[('gas', 'density')].d[idx]
        Etot = (rhoE * 4.*np.pi/3. * ((r+dr)**3 - r**3)).sum()
        etot = (rhoe * 4.*np.pi/3. * ((r+dr)**3 - r**3)).sum()
        Mtot = (rho * 4.*np.pi/3. * ((r+dr)**3 - r**3)).sum()
        print(ds.current_time.d, Etot, etot, Mtot, file=datfile)
        
        # plt.plot(r, rhoE * 4.*np.pi/3. * ((r+dr)**3 - r**3), label=f"t = {ds.current_time.d:.0f} s")

# plt.legend()
# plt.show()
