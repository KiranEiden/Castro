#!/usr/bin/env python3

import h5py
import argparse
import numpy as np

from pynucastro import Nucleus
from collections import defaultdict
import matplotlib.pyplot as plt

#################################
# Utility classes and functions #
#################################

class SliceMaker:
    """Can subscript to generate slices and tuples of slices, since __getitem__ returns the
    input."""
    
    def __getitem__(self, slc):
        
        if isinstance(slc, int):
            return slice(slc, slc+1, 1)
        if isinstance(slc, tuple):
            return tuple(self[item] for item in slc)
        return slc
        
sm = SliceMaker()

def is_in_slice(val, slc):
    
    return ((slc.start is None) or (val >= slc.start)) and ((slc.stop is None) or (val < slc.stop))

def is_in_slices(Z, A, slc_tup):

    Zslc = slc_tup[0]
    Aslc = slc_tup[1]
    
    return is_in_slice(Z, Zslc) and is_in_slice(A, Aslc)
    
def is_in_slice_list(Z, A, slc_list):
    
    return any(is_in_slices(Z, A, slc_tup) for slc_tup in slc_list)
    
def arr_maker(nx, ny):
    
    def make_arr():
        return np.zeros((nx, ny))
    
    return make_arr
    
######################
# Bin configurations #
######################
    
"""
Format: dict of dicts of lists
 - Keys for top-level dict are string identifiers
 - Keys for second-level dicts are Nucleus objects
 - Lists consist of tuples of (Z, A) slices
"""
bin_configs = \
{
    "LLH": # light-lanthanide-heavy
    {
        Nucleus('H1'): [sm[:57, :]],
        Nucleus('La114'): [sm[57:72, :]],
        Nucleus('Hf144'): [sm[72:, :]]
    }
}

if __name__ == "__main__":
    
    #########################
    # Argument parser setup #
    #########################

    description = """Create a new model file with reduced dimensionality in the composition, where
        elements from the input file are binned into new elements in the output file. The auxiliary
        quantities Abar (mean molecular weight) and the electron fraction Y_e are also included."""
    model_file_help = "Input model file to convert."
    bin_config_help = "String specifying bin configuration for binning procedure."
    outfile_help = "Path of output model file."
    ambient_spec_help = "Species to use for ambient medium."
    ambient_dens_thresh_help = """Density threshold to use to determine if a cell is an ambient
            cell. Any cells with a density less than the threshold will be treated as part of the
            ambient medium."""
    wind_spec_help = "Species to use for engine wind."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model_file", help=model_file_help)
    parser.add_argument('-bc', "--bin_config", default="LLH", help=bin_config_help)
    parser.add_argument('-o', "--outfile", default="binned_model.h5", help=outfile_help)
    parser.add_argument("--ambient_spec", default="h2", type=Nucleus, help=ambient_spec_help)
    parser.add_argument("--ambient_dens_thresh", default=1e-39, type=float,
        help=ambient_dens_thresh_help)
    parser.add_argument("--wind_spec", default="h3", type=Nucleus, help=wind_spec_help)

    args = parser.parse_args()

    #################
    # Do conversion #
    #################
    
    fin = h5py.File(args.model_file)
    bc = bin_configs[args.bin_config]
    
    Zin = fin["Z"][()]
    Ain = fin["A"][()]
    comp = fin["comp"][()]
    rho = fin["rho"][()]
    
    # Normalize composition
    for ispec in range(len(Zin)):
        comp[:, :, ispec] /= comp.sum(axis=2)
    
    # Will be keyed by nuclei in bin config
    binned_comp = defaultdict(arr_maker(*comp.shape[:2]))
    
    for ispec in range(len(Zin)):
        
        spec_binned = False
        
        for bin_nuc, bins in bc.items():
            if is_in_slice_list(Zin[ispec], Ain[ispec], bins):
                assert not spec_binned, "There should be no overlap in bins in bin configuration."
                binned_comp[bin_nuc] += comp[:, :, ispec]
                spec_binned = True
                
        assert spec_binned, "All species in model file must be fall within a bin."
        
    Abar = 1. / (comp / Ain).sum(axis=2)
    Y_e = (Zin * comp / Ain).sum(axis=2)
    
    # Now adjust composition in ambient region
    amb_mask = rho < args.ambient_dens_thresh
    
    # The ambient medium is treated as a separate species
    assert args.ambient_spec not in bc, """Ambient medium should be treated as a separate species
            from the species we bin into."""
    for arr in binned_comp.values():
        arr[amb_mask] = 0.0
        
    Abar[amb_mask] = args.ambient_spec.A
    Y_e[amb_mask] = args.ambient_spec.Z / args.ambient_spec.A
    
    # Construct new Z, A, and composition arrays
    nucs = [args.ambient_spec] + sorted(bc.keys()) + [args.wind_spec]
    bin_nucs = nucs[1:-1]
    
    fout = h5py.File(args.outfile, 'w')
    excl = {'Z', 'A', 'comp'}
    for field in fin.keys():
        if field not in excl:
            fout.create_dataset(field, data=fin[field][()], dtype='d')
    fout.create_dataset('Z', data=[n.Z for n in nucs], dtype='i')
    fout.create_dataset('A', data=[n.A for n in nucs], dtype='i')
    
    comp_out = np.zeros((*Abar.shape, len(nucs)))
    comp_out[amb_mask, 0] = 1.0
    for i, nuc in enumerate(bin_nucs):
        comp_out[:, :, i+1] = binned_comp[nuc]
    fout.create_dataset('comp', data=comp_out, dtype='d')
    
    fout.create_dataset('Abar', data=Abar, dtype='d')
    fout.create_dataset('Y_e', data=Abar, dtype='d')
    
    fin.close()
    fout.close()
    
