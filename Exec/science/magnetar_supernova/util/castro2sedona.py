#!/usr/bin/env python3

import yt
import sys
import numpy as np
import h5py
import argparse
import analysis_util as au

arad = 7.5657e-15

#########################
# Argument parser setup #
#########################

description = "Convert Castro plotfiles to Sedona model files."
datasets_help = "List of Castro plotfiles to convert."
level_help = "AMR level associated with the uniform grid to use in the Sedona model files."

parser = argparse.ArgumentParser(description=description)
parser.add_argument('datasets', nargs='*', help=datasets_help)
parser.add_argument('-l', '--level', nargs='+', type=int, default=0, help=level_help)

args = parser.parse_args()

##############
# Load files #
##############

ts = args.datasets
if len(ts) < 1:
    sys.exit("No files were available to be loaded.")

print("Will load the following files: {}\n".format(ts))

tf = lambda file: yt.load(file.rstrip('/'), hint='CastroDataset')
ts = list(map(tf, ts))

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

# A few default atomic weights
def_A = {'h': 1, 'he': 4, 'c': 12, 'o': 16, 'si': 28, 'ni': 56}

def pad_spec_name(spec_name):
    
    if not spec_name[-1].isnumeric():
        return spec_name + str(def_A[spec_name])
    return spec_name

spec_names = map(lambda s: s[2:-1].lower(), mfrac_fields)
spec_names = map(pad_spec_name, spec_names)
nuclides = list(map(au.Nuclide, spec_names))

idx = sorted(range(len(nuclides)), key=lambda i: nuclides[i])
nuclides = [nuclides[i] for i in idx]
mfrac_fields = [mfrac_fields[i] for i in idx]

##########################################
# Loop and convert to Sedona model files #
##########################################

for ds in ts:
    
    assert ds.dimensionality == 2, "Only support 2d datasets currently"
    
    ad = au.AMRData(ds, args.level)
    
    fout = h5py.File(f'model_{ds}.h5', 'w')
    fout.create_dataset('time', data=[ds.current_time.d], dtype='d')
    fout.create_dataset('rmin', data=ad.left_edge.d, dtype='d')
    fout.create_dataset('dr', data=ad.dds[:, args.level].d, dtype='d')
    fout.create_dataset('rho', data=ad.field_data('density', units=False), dtype='d')
    erad = ad.field_data('pressure', units=False) * 3.
    fout.create_dataset('temp', data=(erad / arad)**0.25, dtype='d')
    fout.create_dataset('erad', data=erad, dtype='d')
    fout.create_dataset('vx', data=ad.field_data('x_velocity', units=False), dtype='d')
    fout.create_dataset('vz', data=ad.field_data('y_velocity', units=False), dtype='d')
    fout.create_dataset('Z', data=[n.Z for n in nuclides], dtype='i')
    fout.create_dataset('A', data=[n.A for n in nuclides], dtype='i')
    
    comp = np.zeros((len(nuclides), *ad.ncells[:, args.level]), dtype=np.float64)
    for i in range(len(nuclides)):
        comp[i, ...] = ad.field_data(mfrac_fields[i], units=False)
    comp = np.transpose(comp, axes=(*range(1, ad.dim+1), 0))
    fout.create_dataset('comp', data=comp, dtype='d')
    
    fout.close()
