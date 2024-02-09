#!/usr/bin/env python3

import re
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

class Nuclide:
    
    long_pat = re.compile("(?P<name>[A-Za-z]+)-(?P<A>[0-9]+)")
    short_pat = re.compile("(?P<symbol>[A-Za-z]{1,3})(?P<A>[0-9]+)")
    
    elements = \
    {
        'hydrogen': 1,
        'helium': 2,
        'lithium': 3,
        'beryllium': 4,
        'boron': 5,
        'carbon': 6,
        'nitrogen': 7,
        'oxygen': 8,
        'fluorine': 9,
        'neon': 10,
        'sodium': 11,
        'magnesium': 12,
        'aluminum': 13,
        'silicon': 14,
        'phosphorus': 15,
        'sulfur': 16,
        'chlorine': 17,
        'argon': 18,
        'potassium': 19,
        'calcium': 20,
        'scandium': 21,
        'titanium': 22,
        'vanadium': 23,
        'chromium': 24,
        'manganese': 25,
        'iron': 26,
        'cobalt': 27,
        'nickel': 28,
        'copper': 29,
        'zinc': 30,
        'gallium': 31,
        'germanium': 32,
        'arsenic': 33,
        'selenium': 34,
        'bromine': 35,
        'krypton': 36,
        'rubidium': 37,
        'strontium': 38,
        'yttrium': 39,
        'zirconium': 40,
        'niobium': 41,
        'molybdenum': 42,
        'technetium': 43,
        'ruthenium': 44,
        'rhodium': 45,
        'palladium': 46,
        'silver': 47,
        'cadmium': 48,
        'indium': 49,
        'tin': 50,
        'antimony': 51,
        'tellurium': 52,
        'iodine': 53,
        'xenon': 54,
        'cesium': 55,
        'barium': 56,
        'lanthanum': 57,
        'cerium': 58,
        'praseodymium': 59,
        'neodymium': 60,
        'promethium': 61,
        'samarium': 62,
        'europium': 63,
        'gadolinium': 64,
        'terbium': 65,
        'dysprosium': 66,
        'holmium': 67,
        'erbium': 68,
        'thulium': 69,
        'ytterbium': 70,
        'lutetium': 71,
        'hafnium': 72,
        'tantalum': 73,
        'tungsten': 74,
        'rhenium': 75,
        'osmium': 76,
        'iridium': 77,
        'platinum': 78,
        'gold': 79,
        'mercury': 80,
        'thallium': 81,
        'lead': 82,
        'bismuth': 83,
        'polonium': 84,
        'astatine': 85,
        'radon': 86,
        'francium': 87,
        'radium': 88,
        'actinium': 89,
        'thorium': 90,
        'protactinium': 91,
        'uranium': 92,
        'neptunium': 93,
        'plutonium': 94,
        'americium': 95,
        'curium': 96,
        'berkelium': 97,
        'californium': 98,
        'einsteinium': 99,
        'fermium': 100,
        'mendelevium': 101,
        'nobelium': 102,
        'lawrencium': 103,
        'rutherfordium': 104,
        'dubnium': 105,
        'seaborgium': 106,
        'bohrium': 107,
        'hassium': 108,
        'meitnerium': 109,
        'darmstadtium': 110,
        'roentgenium': 111,
        'copernicium': 112,
        'nihonium': 113,
        'flerovium': 114,
        'moscovium': 115,
        'livermorium': 116,
        'tennessine': 117,
        'oganesson': 118
    }
    
    sym_to_name = \
    {
        'h': 'hydrogen',
        'he': 'helium',
        'li': 'lithium',
        'be': 'beryllium',
        'b': 'boron',
        'c': 'carbon',
        'n': 'nitrogen',
        'o': 'oxygen',
        'f': 'fluorine',
        'ne': 'neon',
        'na': 'sodium',
        'mg': 'magnesium',
        'al': 'aluminum',
        'si': 'silicon',
        'p': 'phosphorus',
        's': 'sulfur',
        'cl': 'chlorine',
        'ar': 'argon',
        'k': 'potassium',
        'ca': 'calcium',
        'sc': 'scandium',
        'ti': 'titanium',
        'v': 'vanadium',
        'cr': 'chromium',
        'mn': 'manganese',
        'fe': 'iron',
        'co': 'cobalt',
        'ni': 'nickel',
        'cu': 'copper',
        'zn': 'zinc',
        'ga': 'gallium',
        'ge': 'germanium',
        'as': 'arsenic',
        'se': 'selenium',
        'br': 'bromine',
        'kr': 'krypton',
        'rb': 'rubidium',
        'sr': 'strontium',
        'y': 'yttrium',
        'zr': 'zirconium',
        'nb': 'niobium',
        'mo': 'molybdenum',
        'tc': 'technetium',
        'ru': 'ruthenium',
        'rh': 'rhodium',
        'pd': 'palladium',
        'ag': 'silver',
        'cd': 'cadmium',
        'in': 'indium',
        'sn': 'tin',
        'sb': 'antimony',
        'te': 'tellurium',
        'i': 'iodine',
        'xe': 'xenon',
        'cs': 'cesium',
        'ba': 'barium',
        'la': 'lanthanum',
        'ce': 'cerium',
        'pr': 'praseodymium',
        'nd': 'neodymium',
        'pm': 'promethium',
        'sm': 'samarium',
        'eu': 'europium',
        'gd': 'gadolinium',
        'tb': 'terbium',
        'dy': 'dysprosium',
        'ho': 'holmium',
        'er': 'erbium',
        'tm': 'thulium',
        'yb': 'ytterbium',
        'lu': 'lutetium',
        'hf': 'hafnium',
        'ta': 'tantalum',
        'w': 'tungsten',
        're': 'rhenium',
        'os': 'osmium',
        'ir': 'iridium',
        'pt': 'platinum',
        'au': 'gold',
        'hg': 'mercury',
        'tl': 'thallium',
        'pb': 'lead',
        'bi': 'bismuth',
        'po': 'polonium',
        'at': 'astatine',
        'rn': 'radon',
        'fr': 'francium',
        'ra': 'radium',
        'ac': 'actinium',
        'th': 'thorium',
        'pa': 'protactinium',
        'u': 'uranium',
        'np': 'neptunium',
        'pu': 'plutonium',
        'am': 'americium',
        'cm': 'curium',
        'bk': 'berkelium',
        'cf': 'californium',
        'es': 'einsteinium',
        'fm': 'fermium',
        'md': 'mendelevium',
        'no': 'nobelium',
        'lr': 'lawrencium',
        'rf': 'rutherfordium',
        'db': 'dubnium',
        'sg': 'seaborgium',
        'bh': 'bohrium',
        'hs': 'hassium',
        'mt': 'meitnerium',
        'ds': 'darmstadtium',
        'rg': 'roentgenium',
        'cn': 'copernicium',
        'nh': 'nihonium',
        'fl': 'flerovium',
        'mc': 'moscovium',
        'lv': 'livermorium',
        'ts': 'tennessine',
        'og': 'oganesson'
     }
    
    name_to_sym = {v: k for k, v in sym_to_name.items()}
    
    def __init__(self, string):
        
        try:
            
            string = string.strip()
            assert self.isnuclide(string)
            
            match = self.match_long(string)
            
            if match:
                
                self.name, self.A = match.groups()
                self.name = self.name.lower()
                self.sym = self.name_to_sym[self.name]
                
            else:
                
                match = self.match_short(string)
                self.sym, self.A = match.groups()
                self.sym = self.sym.lower()
                self.name = self.sym_to_name[self.sym]
                
            self.A = int(self.A)
            self.Z = self.elements[self.name]
            self.N = self.A - self.Z
            
            assert self.N >= 0
            
        except (ValueError, KeyError, AssertionError):
            
            raise ValueError("Invalid nuclide string: '{}'.".format(string)) from None
    
    def __repr__(self):
        
        return "Nuclide({}, N={}, Z={})".format(self, *self)
        
    def __str__(self):
        
        return self.short_str().capitalize()
        
    def __iter__(self):
        
        yield self.Z
        yield self.N
        
    def _compare(self, other, op):
        
        if isinstance(other, Nuclide):
            return op(tuple(self), tuple(other))
        else:
            try:
                other_Z, other_N = other
                return op(tuple(self), (other_Z, other_N))
            except (TypeError, ValueError):
                raise NotImplementedError
    
    def __eq__(self, other):
        
        return self._compare(other, lambda a, b: a == b)
        
    def hash(self):
        
        return hash(tuple(self))
        
    def __lt__(self, other):
        
        return self._compare(other, lambda a, b: a < b)
        
    def __gt__(self, other):
        
        return self._compare(other, lambda a, b: a > b)
        
    def __le__(self, other):
        
        return self._compare(other, lambda a, b: a <= b)
    
    def __ge__(self, other):
        
        return self._compare(other, lambda a, b: a >= b)
        
    def short_str(self):
        
        return self.sym + str(self.A)
        
    def long_str(self):
        
        return self.name + "-" + str(self.A)
            
    @classmethod
    def isnuclide(cls, string):
        
        return cls.match_long(string) or cls.match_short(string)
        
    @classmethod
    def match_long(cls, string):
        
        return cls.long_pat.match(string)
        
    @classmethod
    def match_short(cls, string):
        
        return cls.short_pat.match(string)

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
nuclides = list(map(Nuclide, spec_names))

idx = sorted(range(len(nuclides)), key=lambda i: nuclides[i])
nuclides = [nuclides[i] for i in idx]
mfrac_fields = [mfrac_fields[i] for i in idx]

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
