
import yt
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
    
    def __init__(self, files, use_mpi=False):
        
        try:
            len(files)
        except TypeError:
            files = list(files)
        
        self.files = files
        self.use_mpi = use_mpi
        
        if self.use_mpi:
            self.MPI = mpi_importer()
            self.comm = self.MPI.COMM_WORLD
            
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
            
    def serial_generator(self):
        
        for f in self.files:
            yield self._load(f)
            
    def parallel_generator(self):
        
        MPI_N = self.comm.Get_size()
        MPI_rank = self.comm.Get_rank()
        
        for i in range(MPI_rank, len(self.files), MPI_N):
            yield self._load(self.files[i])
            
    def parallel_enumerator(self):
        
        MPI_N = self.comm.Get_size()
        MPI_rank = self.comm.Get_rank()
        
        for i in range(MPI_rank, len(self.files), MPI_N):
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
            
            idx[d] = np.rint((bounds[d] - lo) / dd - 0.5).astype(np.int32)
            bounds[d] = (idx[d] + 0.5) * dd + lo
            
        return idx, bounds
        
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
    
    for i, field in enumerate(fields):
        color_iter = iter(mpl_colors)
        for r, th, data in get_prof_2d(ds, nrays, field, **kwargs):
            plt.plot(r, data, label=f"θ = {th*180/np.pi}°",
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
        
    
def get_avg_prof_2d(ds, nrays=100, r=None, z=None, data=None, field=None, level=0,
        ang_range=(0.0, np.pi), return_minmax=False, return_r=False):
    
    if settings['verbose'] and (field is not None):
        print(f"Getting angle-averaged {field} profile for {ds}...")
        
    if (r is None) or (z is None) or (data is None):
        ad = AMRData(ds, level)
        if (r is None) or (z is None):
            r, z = ad.position_data(units=False)
        if data is None:
            data = ad.field_data(field, units=False)
        
    if isinstance(r, u.unyt_array):
        r = r.d
    if isinstance(z, u.unyt_array):
        z = z.d
    if isinstance(data, u.unyt_array):
        data = data.d
    
    # Get 1d list of r values
    r1d = r[:,0]
    
    # Fixed resolution rays don't work in 2d with my yt version
    interp = RegularGridInterpolator((r1d, z[0]), data, bounds_error=False, fill_value=None)
    theta = np.linspace(*ang_range, num=nrays)
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
        pts = interp(r1d[i] * xi)
        update()
        
    if not (return_minmax or return_r):
        return avg
        
    return_items = (avg,)
    if return_minmax:
        return_items += (mnm, mxm)
    if return_r:
        return_items += (r1d,)
    return return_items
