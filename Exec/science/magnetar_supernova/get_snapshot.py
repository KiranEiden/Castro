#!/usr/bin/env python3

import yt
import sys
import argparse
import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from analysis_util import AMRData

parser = argparse.ArgumentParser()
parser.add_argument('datafiles', nargs="*")
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('-f', "--fields", nargs="*", default=['density', 'magvel', 'pressure'])
parser.add_argument('-o', '--output', default=None)
args = parser.parse_args()

ts = args.datafiles
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
ts = map(tf, ts)
    
def calc_2d(ds):
    
    # Make data object and retrieve data
    ad = AMRData(ds, args.level, verbose=True)
    r, z = ad.position_data()
    field_data = dict()
    for field in args.fields:
        field_data[field] = ad[field]
    
    # Write data to file
    if not args.output:
        args.output = f"snapshot_{ds}.npz"
    print("Saving snapshot...")
    np.savez_compressed(args.output, r=r, z=z, **field_data)

for ds in ts:
        
    n = ds.dimensionality
    fstr = f"calc_{n}d(ds)"
    vals = eval(fstr)
    
    print("Task completed.")
