#!/usr/bin/env python3

import yt
import sys
import numpy as np
import h5py
import argparse

from gridplot import Nuclide
from yt.frontends.boxlib.data_structures import CastroDataset

a_rad = 7.5657e-15

#########################
# Argument parser setup #
#########################

description = "Convert Castro plotfiles to Sedona model files."
datasets_help = "List of Castro plotfiles to convert."
dim_help = "Dimensions of the uniform grid to use in the Sedona model files."

parser = argparse.ArgumentParser(description=description)
parser.add_argument('datasets', nargs='*', help=datasets_help)
parser.add_argument('-d', '--dim', nargs='+', type=int, help=dim_help)

args = parser.parse_args(sys.argv[1:])

##############
# Load files #
##############

ts = args.datasets
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: CastroDataset(file.rstrip('/'))
ts = list(map(tf, ts))

assert args.dim, "Must specify grid dimension for Sedona model file"
args.dd = np.array(args.dim, np.float64)

def flatten(seq):
    """ Collapse a 2d input down to a 1d input. """
    
    for subseq in seq:
        for item in subseq:
            yield item
 
##################################################
# Setup for determining elements and composition #
##################################################

xfilt = lambda f: f.startswith("X(") and f.endswith(")")
fields = map(lambda f: f[1], ts[0].field_list)
mfrac_fields = list(filter(xfilt, fields))

def_A = {'h': 1, 'he': 4, 'c': 12, 'o': 16}

def pad_spec_name(spec_name):
    
    if not spec_name[-1].isnumeric():
        return spec_name + str(def_A[spec_name])
    return spec_name

spec_names = map(lambda s: s[2:-1].lower(), mfrac_fields)
spec_names = map(pad_spec_name, spec_names)
nuclides = list(map(Nuclide, spec_names))

idx = sorted(range(len(nuclides)), key=lambda i: nuclides[i])
nuclides = [nuclides[i] for i in idx]
mfrac_fields = [mfrac_fields[i] for i in idx]

for ds in ts:
    
    assert ds.dimensionality == 2, "Only support 2d datasets currently"
    
    slc = ds.slice(2, np.pi)
    rmin = ds.domain_left_edge[:ds.dimensionality]
    rmax = ds.domain_right_edge[:ds.dimensionality]
    frb = yt.FixedResolutionBuffer(slc, list(flatten(zip(rmin, rmax))), args.dim)
    
    fout = h5py.File(f'model_{ds}.h5', 'w')
    fout.create_dataset('time', data=[ds.current_time.d], dtype='d')
    fout.create_dataset('rmin', data=rmin.d, dtype='d')
    fout.create_dataset('dr', data=(rmax - rmin).d/args.dd, dtype='d')
    fout.create_dataset('rho', data=frb[('gas', 'density')].d, dtype='d')
    fout.create_dataset('temp', data=(frb[('boxlib', 'pressure')].d*3 / a_rad)**0.25, dtype='d')
    fout.create_dataset('erad', data=frb[('boxlib', 'pressure')].d*3, dtype='d')
    fout.create_dataset('vx', data=frb[('boxlib', 'x_velocity')], dtype='d')
    fout.create_dataset('vz', data=frb[('boxlib', 'z_velocity')], dtype='d')
    fout.create_dataset('Z', data=[n.Z for n in nuclides], dtype='i')
    fout.create_dataset('A', data=[n.A for n in nuclides], dtype='i')
    
    comp = np.zeros((len(nuclides), *args.dim), dtype=np.float64)
    for i in range(len(nuclides)):
        comp[i, ...] = frb[mfrac_fields[i]].d
    comp = np.transpose(comp, axes=(*range(1, ds.dimensionality+1), 0))
    fout.create_dataset('comp', data=comp, dtype='d')
    
    fout.close()
