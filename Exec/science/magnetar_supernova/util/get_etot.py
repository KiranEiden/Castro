#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import unyt as u
import analysis_util as au

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('-o', '--output', default="energy.dat")
args = parser.parse_args()

au.settings['verbose'] = True
    
ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
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
    
    # Make data object and retrieve data
    ad = au.AMRData(ds, args.level)
    r, z = ad.position_data(units=False)
    dr, dz = ad.dds[:, args.level].d
    vol = np.pi * ((r+dr/2)**2 - (r-dr/2)**2) * dz
    
    rhoE = ad['rho_E'].d
    rhoe = ad['rho_e'].d
    rho = ad['density'].d

    # Calculate total energy, internal energy, and mass
    print("Calculating derived quantities...")
    Etot = (rhoE * vol).sum()
    etot = (rhoe * vol).sum()
    Mtot = (rho * vol).sum()
    
    return Etot, etot, Mtot

with open(args.output, 'w') as datfile:
    
    print(f"Data file: {args.output}.")
    
    print('#', 't', 'E', 'e', 'M', file=datfile)
    print('#', 's', 'erg', 'erg', 'g', file=datfile)

    for ds in ts:
        
        n = ds.dimensionality
        fstr = f"calc_{n}d(ds)"
        Etot, etot, Mtot = eval(fstr)
        
        print(ds.current_time.d, Etot, etot, Mtot, file=datfile)
        
    print("Task completed.")
