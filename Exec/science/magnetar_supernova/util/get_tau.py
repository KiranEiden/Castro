#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import unyt as u
import analysis_util as au

from scipy.integrate import simpson

cs_thom = 6.65e-24 * u.cm**2
amu = 1.660539e-24 * u.g

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument("--use_mpi", action="store_true")
args = parser.parse_args()

ts = sys.argv[1:]
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")
    
if args.use_mpi:
    MPI = au.mpi_importer()
is_main_proc = (not args.use_mpi) or (MPI.COMM_WORLD.Get_rank() == 0)
if is_main_proc:
    print("Will load the following files: {}\n".format(ts))

if args.use_mpi:
    ts = au.FileLoader(ts, True)
    MPI.COMM_WORLD.Barrier()
else:
    tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
    ts = map(tf, ts)

for ds in ts:

    with open(f'tau_{ds}.dat', 'w') as datfile:

        print('# theta', 'N', 'tau', file=datfile)
        print('# rad', 'cm**-2', 'dimensionless', file=datfile)
        print('# time (s):', ds.current_time.d, file=datfile)
        
        for r, th, rho in au.get_prof_2d(ds, 500, 'density'):
            
            col_dens = simpson((rho / amu).d, r.d) * u.cm**-2
            tau = col_dens * cs_thom
            
            print(th, col_dens.d, tau.d, file=datfile)
