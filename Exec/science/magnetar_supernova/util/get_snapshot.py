#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
import analysis_util as au

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('-f', "--fields", nargs="*", default=['density', 'x_velocity', 'y_velocity', 'pressure', 'entropy', 'abar'])
parser.add_argument('-wt', '--write_times', action='store_true')
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

ts = au.FileLoader(ts)
    
def calc_2d(ds):
    
    # Make data object and retrieve data
    ad = au.AMRData(ds, args.level)
    r, z = ad.position_data(units=False)
    field_data = dict()
    for field in args.fields:
        field_data[field] = ad[field].d
    
    # Write data to file
    outfile = f"snapshot_{ds}.npz"
    print("Saving snapshot...")
    np.savez_compressed(outfile, r=r, z=z, **field_data)

if args.write_times:
    f = open("times.dat", 'w')
    print("# dataset time", file=f)

for ds in ts:
        
    n = ds.dimensionality
    fstr = f"calc_{n}d(ds)"
    eval(fstr)

    if args.write_times:
        print(ds, ds.current_time.d, file=f)
        
if args.write_times:
    f.close()

print("Task completed.")
