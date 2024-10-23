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
add_decay_prod_help = "If provided, will add Co56 and Fe56 to the composition if not already present."
split_elem_help = """Split elements into themselves and one other element. Can supply any number elements to split
        (including repeats) using the format base_element:product_element:product_element_fraction, where the last
        component will be assumed to be 0.5 if omitted. Splitting operations will be done in order."""
H_to_lodders_help = "Load 'lodders_massfrac.dat' and redistribute hydrogen accordingly (only loads even elements)."
convert_elem_help = """Convert first comma-separated sequence of elements to even split of second sequence. The
        first and second sequence should be separated by a colon. Can supply multiple items to do multiple
        conversions."""
t0_help = "The initial time offset of the Castro simulation, in seconds."
tscale_help = """Ratio of engine timescale to use for the output to engine timescale in the Castro simulation.
        If tscale != 1.0, the time, lengthscales, density, etc. will be rescaled to approximate the result of
        a Castro simulation with the output engine timescale, assuming the engine-to-ejecta energy ratio is
        held constant."""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('datasets', nargs='*', help=datasets_help)
parser.add_argument('-l', '--level', type=int, default=0, help=level_help)
parser.add_argument('--add_decay_prod', action='store_true', help=add_decay_prod_help)
parser.add_argument('--split_elem', nargs='+', help=split_elem_help)
parser.add_argument('--H_to_lodders', action='store_true', help=H_to_lodders_help)
parser.add_argument('--convert_elem', nargs='+', help=convert_elem_help)
parser.add_argument('--t0', type=float, default=0.0, help=t0_help)
parser.add_argument('--tscale', type=float, default=1.0, help=tscale_help)

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
            
class MassFracTransform:
    
    def __init__(self, func):
        
        self.func = func
        
    def __call__(self, ad):
        
        return self.func(ad)
        
    def __add__(self, other):
        
        return self.__class__(lambda ad: self(ad) + other(ad))
        
    def __mul__(self, other):
        
        return self.__class__(lambda ad: self(ad) * other(ad))
        
    @classmethod
    def mfrac_accessor(cls, field):
    
        def get_mfrac(ad):
            return ad.field_data(field, units=False)
        return cls(get_mfrac)
    
    @classmethod
    def float_mapping(cls, f=0.0):
        
        return cls(lambda x: f)
 
##################################################
# Setup for determining elements and composition #
##################################################

xfilt = lambda f: f.startswith("X(") and f.endswith(")")
fields = map(lambda f: f[1], ts[0].field_list)
mfrac_fields = list(filter(xfilt, fields))
mfrac_ops = dict()

# A few default atomic weights
def_A = {'h': 1, 'he': 4, 'c': 12, 'o': 16, 'si': 28, 'ni': 56}

def pad_spec_name(spec_name):
    
    if not spec_name[-1].isnumeric():
        return spec_name + str(def_A[spec_name])
    return spec_name

spec_names = map(lambda s: s[2:-1].lower(), mfrac_fields)
spec_names = map(pad_spec_name, spec_names)
nuclides = list(map(au.Nuclide, spec_names))

for i in range(len(nuclides)):
    mfrac_ops[nuclides[i]] = MassFracTransform.mfrac_accessor(mfrac_fields[i])

if args.split_elem:

    for item in args.split_elem:
    
        if item.count(':') == 2:
            base_elem, prod_elem, frac_conv = item.split(':')
            frac_conv = float(frac_conv)
        else:
            base_elem, prod_elem = item.split(':')
            frac_conv = 0.5
            
        base_elem = au.Nuclide(base_elem)
        prod_elem = au.Nuclide(prod_elem)

        i_base = nuclides.index(base_elem)

        if mfrac_fields[i_base] is not None:
            prod_op = MassFracTransform.mfrac_accessor(mfrac_fields[i_base])
        else:
            prod_op = mfrac_ops[base_elem]
        prod_op *= MassFracTransform.float_mapping(frac_conv)
            
        if prod_elem not in mfrac_ops:
            nuclides.append(prod_elem)
            mfrac_fields.append(None)
            mfrac_ops[prod_elem] = prod_op
        else:
            mfrac_ops[prod_elem] += prod_op
        mfrac_ops[base_elem] *= MassFracTransform.float_mapping(1.0 - frac_conv)
    
if args.H_to_lodders:
    
    lodders_Z = []
    lodders_mfrac = []
    
    with open('lodders_massfrac.dat', 'r') as file:
        for i, line in enumerate(file):
            if (i != 0) and (i % 2 == 0):
                continue # Keep only even elements and hydrogen
            Z, mfrac = line.strip().split()
            if int(Z) > 28:
                break
            lodders_Z.append(int(Z))
            lodders_mfrac.append(float(mfrac))
    lodders_mfrac = np.array(lodders_mfrac)
    
    lodders_mfrac /= lodders_mfrac.sum()
    i_H = nuclides.index((1, 0))
    for i, Z in enumerate(lodders_Z):
        if Z == 1:
            elem = au.Nuclide((1, 0))
            mfrac_ops[elem] *= MassFracTransform.float_mapping(lodders_mfrac[i])
        else:
            elem = au.Nuclide((Z, Z))
            if elem not in mfrac_ops:
                nuclides.append(elem)
                mfrac_fields.append(None)
                mfrac_ops[elem] = MassFracTransform.float_mapping()
            elem_op = MassFracTransform.mfrac_accessor(mfrac_fields[i_H])
            elem_op *= MassFracTransform.float_mapping(lodders_mfrac[i])
            mfrac_ops[elem] += elem_op

if args.convert_elem:
    
    for pair in args.convert_elem:
    
        elems = pair.split(':')
        elems = [list(map(au.Nuclide, seq.split(','))) for seq in elems]
        mfrac_sum_acc = sum((mfrac_ops[elem] for elem in elems[0]), start=MassFracTransform.float_mapping())
    
        for elem in elems[1]:
            if elem not in mfrac_ops:
                nuclides.append(elem)
                mfrac_fields.append(None)
                mfrac_ops[elem] = MassFracTransform.float_mapping()
            mfrac_ops[elem] += mfrac_sum_acc * MassFracTransform.float_mapping(1. / len(elems[1]))
    
        for elem in elems[0]:
            i_elem = nuclides.index(elem)
            del nuclides[i_elem]
            del mfrac_fields[i_elem]
            del mfrac_ops[elem]

if args.add_decay_prod:

    Fe56 = au.Nuclide('Fe56')
    Co56 = au.Nuclide('Co56')

    if Fe56 not in mfrac_ops:
        nuclides.append(Fe56)
        mfrac_ops[Fe56] = MassFracTransform.float_mapping()
    if Co56 not in mfrac_ops:
        nuclides.append(Co56)
        mfrac_ops[Co56] = MassFracTransform.float_mapping()

idx = sorted(range(len(nuclides)), key=lambda i: nuclides[i])
nuclides = [nuclides[i] for i in idx]

##########################################
# Loop and convert to Sedona model files #
##########################################

for ds in ts:
    
    assert ds.dimensionality == 2, "Only support 2d datasets currently"
    
    ad = au.AMRData(ds, args.level)
    
    tscale = args.tscale
    rscale = args.tscale
    rhoscale = 1. / args.tscale**3
    escale = 1. / args.tscale**3
    
    fout = h5py.File(f'model_{ds}.h5', 'w')
    fout.create_dataset('time', data=[(ds.current_time.d + args.t0)*tscale], dtype='d')
    fout.create_dataset('rmin', data=ad.left_edge.d*rscale, dtype='d')
    fout.create_dataset('dr', data=ad.dds[:, args.level].d*rscale, dtype='d')
    fout.create_dataset('rho', data=ad.field_data('density', units=False)*rhoscale, dtype='d')
    erad = ad.field_data('pressure', units=False) * 3.
    fout.create_dataset('temp', data=(erad / arad)**0.25, dtype='d')
    fout.create_dataset('erad', data=erad*escale, dtype='d')
    fout.create_dataset('vx', data=ad.field_data('x_velocity', units=False), dtype='d')
    fout.create_dataset('vz', data=ad.field_data('y_velocity', units=False), dtype='d')
    fout.create_dataset('Z', data=[n.Z for n in nuclides], dtype='i')
    fout.create_dataset('A', data=[n.A for n in nuclides], dtype='i')
    
    comp = np.empty((len(nuclides), *ad.ncells[:, args.level]), dtype=np.float64)
    for i in range(len(nuclides)):
        comp[i, ...] = mfrac_ops[nuclides[i]](ad)
    comp = np.transpose(comp, axes=(*range(1, ad.dim+1), 0))
    fout.create_dataset('comp', data=comp, dtype='d')
    
    fout.close()
