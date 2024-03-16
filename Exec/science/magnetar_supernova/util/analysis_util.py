
import yt
import re
import unyt as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

prop_cycle = plt.rcParams['axes.prop_cycle']
mpl_colors = prop_cycle.by_key()['color']

settings = dict(verbose=False)

#############
# MPI Stuff #
#############

class MPIImportError(Exception):
    pass


class FailedMPIImport:
    """Class that can replace an mpi4py.MPI import and will throw an error if used."""

    def __init__(self, error=None, msg=None):

        self.error_obj = error

        if msg is None:
            msg = ("Failed to import MPI from mpi4py. Check your mpi4py installation if you"
                   " want to run with MPI.")
        self.msg = msg

    def __getattr__(self, attr):

        if self.error_obj is not None:
            raise MPIImportError(self.msg) from self.error_obj
        raise MPIImportError(self.msg)


def mpi_importer():
    """
    Lazy MPI import, where we only throw an error if the import failed and then we attempt to use
    the object.
    """

    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except (ModuleNotFoundError, ImportError) as e:
        MPI = FailedMPIImport(e)

    return MPI

##############################################
# Parellelizable Data File Loader and Writer #
##############################################

class FileLoader:
    
    def __init__(self, files, use_mpi=False, def_decomp='cyclic'):
        
        try:
            len(files)
        except TypeError:
            files = list(files)
        
        self.files = files
        self.use_mpi = use_mpi
        
        if self.use_mpi:
            self.MPI = mpi_importer()
            self.comm = self.MPI.COMM_WORLD
            
        assert def_decomp in ('cyclic', 'block')
        self.def_decomp = def_decomp

            
    def __len__(self):
        
        return len(self.files)

        
    def __iter__(self):
        
        if self.use_mpi:
            return self.parallel_generator()
        else:
            return self.serial_generator()

    
    @staticmethod        
    def _load(f):
        
        return yt.load(f, hint='CastroDataset')

               
    def do_decomp(self, decomp_type=None):
        
        MPI_N = self.comm.Get_size()
        MPI_rank = self.comm.Get_rank()
        
        if decomp_type is None:
            decomp_type = self.def_decomp
        
        if decomp_type == 'block':
            num_in_block = np.full((MPI_N,), len(self.files) // MPI_N, dtype=np.int32)
            num_in_block[:(len(self.files) % MPI_N)] += 1
            start = num_in_block[:MPI_rank].sum()
            stop = min(start + num_in_block[MPI_rank], len(self.files))
            step = 1
        elif decomp_type == 'cyclic':
            start = MPI_rank
            stop = len(self.files)
            step = MPI_N
            
        return start, stop, step

            
    def serial_generator(self):
        
        for f in self.files:
            yield self._load(f)
            
            
    def parallel_generator(self, decomp_type=None):
        
        start, stop, step = self.do_decomp(decomp_type)
        
        for i in range(start, stop, step):
            yield self._load(self.files[i])
            
            
    def parallel_enumerator(self, decomp_type=None):
        
        start, stop, step = self.do_decomp(decomp_type)
        
        for i in range(start, stop, step):
            yield i, self._load(self.files[i])

################################
# Class for Accessing AMR Data #
################################

class AMRData:
    
    def __init__(self, ds, def_level=0):
        
        self.ds = ds
        self.dim = ds.dimensionality
        self.grids = list(ds.index.get_levels())
        self.nlevels = len(self.grids)
        self.def_level = self._level_check(def_level)
        
        self.left_edge = u.unyt_array(ds.domain_left_edge[:self.dim].d, ds.length_unit)
        self.right_edge = u.unyt_array(ds.domain_right_edge[:self.dim].d, ds.length_unit)
        
        self.dds = u.unyt_array(np.empty((self.dim, self.nlevels)), ds.length_unit)
        for l in range(self.nlevels):
            self.dds[:, l] = self.grids[l][0].dds[:self.dim]
        
        self.coarseness = (self.dds.d.T / self.dds[:, -1].d).T.astype(np.int32)
        self.ref = self.coarseness[:, ::-1]
        
        base_cells = ds.domain_dimensions[:self.dim]
        self.ncells = (base_cells * self.ref.T).T
        
        
    def __getitem__(self, field):
        
        return self.field_data(field)
        

    def _level_check(self, level):
        
        if level is None:
            level = self.def_level
        if (level < 0) or (level >= self.nlevels):
            raise ValueError(f"Level number must be in range [0, {self.nlevels-1}]")
        return level
        

    def _unytq_conv(self, val, to):
        
        if isinstance(val, u.unyt_quantity):
            return val.to(to).d
        else:
            return val

        
    def position_data(self, level=None, units=True):
        
        if settings['verbose']:
            print(f"Retrieving position data for {self.ds}...")
        level = self._level_check(level)
        
        ind = np.indices(self.ncells[:, level])
        
        # Using a generator appears to be more efficient than a single NumPy calculation
        # Could depend on hardware
        for i in range(self.dim):
            arr = (ind[i] + 0.5) * self.dds[i, level].d + self.left_edge[i].d
            if units:
                yield u.unyt_array(arr, self.ds.length_unit)
            else:
                yield arr
        
        
    def field_data(self, field, level=None, units=True):
        
        if settings['verbose']:
            print(f"Retrieving {field} data for {self.ds}...")
        level = self._level_check(level)
            
        data = np.empty(self.ncells[:, level])
        block_idx_tab = dict()
        
        for l in range(level + 1):
            
            mult = self.coarseness[:, l] // self.coarseness[:, level]
            
            for g in self.grids[l]:
                
                # Get indices corresponding to grid in larger array
                g_size_ref = g.shape[:self.dim] * mult
                g_size_ref_tup = tuple(g_size_ref)
                ilo = g.get_global_startindex()[:self.dim] * mult
                ihi = g_size_ref + ilo
                grid_idx = tuple(slice(ilo[i], ihi[i]) for i in range(self.dim))
                
                if g_size_ref_tup in block_idx_tab:
                    block_idx = block_idx_tab[g_size_ref_tup]
                else:
                    block_idx = (np.indices(g_size_ref).T // mult).T
                    block_idx = tuple(block_idx)
                    block_idx_tab[g_size_ref_tup] = block_idx
                
                # Can skip multiplying by child mask since we go in order of increasing level
                grid_data = g[field].squeeze()
                data[grid_idx] = grid_data[block_idx]
                
            block_idx_tab.clear()
            
        if units:
            return u.unyt_array(data, grid_data.units)
        return data

        
    def region_idx(self, *bounds, level=None):
        
        level = self._level_check(level)
        bounds = [self._unytq_conv(b, self.ds.length_unit) for b in bounds]
        bounds = np.array(bounds, dtype=np.float64).reshape((self.dim, 2)) * self.ds.length_unit
        idx = np.empty((self.dim, 2), dtype=np.int32)
        
        for d in range(self.dim):
            
            dd = self.dds[d, level]
            lo = self.left_edge[d]
            hi = self.right_edge[d]
            
            bounds[d][bounds[d] < lo] = lo
            bounds[d][bounds[d] > hi] = hi
            
            idx[d] = np.rint((bounds[d] - lo) / dd - 0.5).astype(np.int32)
            bounds[d] = (idx[d] + 0.5) * dd + lo
            
        # Increment right endpoint so it can be used with slices
        idx[:,1] += 1
        
        return idx, bounds

        
    def select_region(self, data, *bounds, level=None):
        
        level = self._level_check(level)
        assert data.shape == tuple(self.ncells[:,level]), f"Shape of input data {data.shape} must"\
                + f"match resolution at input level (level {level})."
        
        idx, bounds = self.region_idx(*bounds, level=level)
        slices = tuple(slice(*idx[i]) for i in range(self.dim))
        return data[slices], bounds

        
########################
# Additional Functions #
########################

def get_prof_2d(ds, nrays, field, ang_range=(0.0, np.pi), origin=None):
    
    if settings['verbose']:
        print(f"Retrieving {field} profiles for {ds}...")
    
    rlo, zlo, plo = ds.domain_left_edge
    rhi, zhi, phi = ds.domain_right_edge
    zero = 0.0 * ds.length_unit
    
    if origin is None:
        origin = (zero,) * 3
    
    for th in np.linspace(*ang_range, num=nrays):
        
        ray = ds.ray(origin, (rhi*np.sin(th), rhi*np.cos(th), zero))
    
        r = np.sqrt(ray[('index', 'r')]**2 + ray[('index', 'z')]**2)
        idx = np.argsort(r)
    
        r = r[idx]
        data = ray[field][idx]
        yield r, th, data
        
        
def plot_prof_2d(ds, nrays, fields, styles=None, savefile=None, **kwargs):
    
    if isinstance(fields, str):
        fields = [fields]
    if styles is None:
        styles = ['-'] * len(fields)
        
    ylabel = kwargs.pop("ylabel", "")
    xlog = kwargs.pop("xlog", False)
    ylog = kwargs.pop("ylog", False)
    log = kwargs.pop("log", False)
    t0 = kwargs.pop("t0", None)
    
    for i, field in enumerate(fields):
        color_iter = iter(mpl_colors)
        for r, th, data in get_prof_2d(ds, nrays, field, **kwargs):
            if Pmag is not None:
                x = r / (ds.current_time.d + t0)
            else:
                x = r
            plt.plot(x, data, label=f"θ = {th*180/np.pi}°",
                    linestyle=styles[i], color=next(color_iter))
    
    plt.xlabel(r"$\sqrt{r^2 + z^2}$ [cm]")
    if ylabel:
        plt.ylabel(ylabel)
    elif len(field) == 1 and ylabel is not None:
        plt.ylabel(f"{field}")
    if xlog or log:
        plt.xscale("log")
    if ylog or log:
        plt.yscale("log")
    plt.legend()
    
    if savefile is not None:
        plt.savefig(savefile)
        
        
def get_avg_prof_2d(ds, nrays=100, r=None, z=None, data=None, **kwargs):
    
    # Process kwargs
    field = kwargs.get("field", None)
    level = kwargs.get("level", 0)
    weight_field = kwargs.get("weight_field", None)
    weight_data = kwargs.get("weight_data", None)
    ang_range = kwargs.get("ang_range", (0.0, np.pi))
    return_minmax = kwargs.get("return_minmax", False)
    return_r = kwargs.get("return_r", False)
    
    if settings['verbose'] and (field is not None):
        print(f"Getting angle-averaged {field} profile for {ds}...")
        
    get_weight_field = (weight_field is not None) and (weight_data is None)
        
    if (r is None) or (z is None) or (data is None) or get_weight_field:
        ad = AMRData(ds, level)
        if (r is None) or (z is None):
            r, z = ad.position_data(units=False)
        if data is None:
            data = ad.field_data(field, units=False)
        if get_weight_field:
            weight_data = ad.field_data(weight_field, units=False)
            
    if isinstance(r, u.unyt_array):
        r = r.d
    if isinstance(z, u.unyt_array):
        z = z.d
    if isinstance(data, u.unyt_array):
        data = data.d
    if isinstance(weight_data, u.unyt_array):
        weight_data = weight_data.d
        
    theta = np.linspace(*ang_range, num=nrays)
        
    if weight_data is not None:
       return _weighted_avg_prof_2d_helper(theta, r, z, data, weight_data, return_minmax, return_r)
    return _avg_prof_2d_helper(theta, r, z, data, return_minmax, return_r)
    
    
def _avg_prof_2d_helper(theta, r, z, data, return_minmax, return_r):
    
    # Get 1d list of r values
    r1d = r[:,0]
    
    # Fixed resolution rays don't work in 2d with my yt version
    interp = RegularGridInterpolator((r1d, z[0]), data, bounds_error=True)
    xi = np.column_stack((np.sin(theta), np.cos(theta)))
    
    if return_minmax:
        def update():
            avg[i] = pts.mean()
            mnm[i] = pts.min()
            mxm[i] = pts.max()
    else:
        def update():
            avg[i] = pts.mean()
    
    avg = np.empty_like(r1d)
    if return_minmax:
        mnm = np.empty_like(r1d)
        mxm = np.empty_like(r1d)
    for i in range(len(r1d)):
        loc = r1d[i] * xi
        np.clip(loc[:, 0], r1d[0], r1d[-1], out=loc[:, 0])
        np.clip(loc[:, 1], z[0, 0], z[0, -1], out=loc[:, 1])
        pts = interp(loc)
        update()
        
    if not (return_minmax or return_r):
        return avg
        
    return_items = (avg,)
    if return_minmax:
        return_items += (mnm, mxm)
    if return_r:
        return_items += (r1d,)
    return return_items
    
    
def _weighted_avg_prof_2d_helper(theta, r, z, data, weight_data, return_minmax, return_r):
    
    # Get 1d list of r values
    r1d = r[:,0]
    
    # Fixed resolution rays don't work in 2d with my yt version
    interp = RegularGridInterpolator((r1d, z[0]), data, bounds_error=True)
    weight_interp = RegularGridInterpolator((r1d, z[0]), weight_data, bounds_error=True)
    xi = np.column_stack((np.sin(theta), np.cos(theta)))
    
    if return_minmax:
        def update():
            avg[i] = (pts*weights).sum() / weights.sum()
            mnm[i] = pts.min()
            mxm[i] = pts.max()
    else:
        def update():
            avg[i] = (pts*weights).sum() / weights.sum()
    
    avg = np.empty_like(r1d)
    if return_minmax:
        mnm = np.empty_like(r1d)
        mxm = np.empty_like(r1d)
    for i in range(len(r1d)):
        loc = r1d[i] * xi
        np.clip(loc[:, 0], r1d[0], r1d[-1], out=loc[:, 0])
        np.clip(loc[:, 1], z[0, 0], z[0, -1], out=loc[:, 1])
        pts = interp(loc)
        weights = weight_interp(loc)
        update()
        
    if not (return_minmax or return_r):
        return avg
        
    return_items = (avg,)
    if return_minmax:
        return_items += (mnm, mxm)
    if return_r:
        return_items += (r1d,)
    return return_items

#################
# Nuclide Class #
#################
    
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
        
        return "Nuclide({}, Z={}, N={})".format(self, *self)

        
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

        
    def __hash__(self):
        
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
