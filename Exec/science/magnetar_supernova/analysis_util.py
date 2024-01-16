
import yt
import unyt as u
import numpy as np

class AMRData:
    
    def __init__(self, ds, def_level=0, verbose=False):
        
        self.ds = ds
        self.dim = ds.dimensionality
        self.def_level = def_level
        self.verbose = verbose
        self.grids = list(ds.index.get_levels())
        self.nlevels = len(self.grids)
        
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
        
    def position_data(self, level=None, units=True):
        
        if self.verbose:
            print(f"Retrieving position data for {self.ds}...")
        
        if level is None:
            level = self.def_level
            
        ind = np.indices(self.ncells[:, level])
        
        # Using a generator appears to the most efficient at higher resolutions
        for i in range(self.dim):
            arr = (ind[i] + 0.5) * self.dds[i, level].d + self.left_edge[i].d
            if units:
                yield u.unyt_array(arr, self.ds.length_unit)
            else:
                yield arr
        
    def field_data(self, field, level=None, units=True):
        
        if self.verbose:
            print(f"Retrieving {field} data for {self.ds}...")
        
        if level is None:
            level = self.def_level
        if (level < 0) or (level >= self.nlevels):
            raise ValueError(f"Level number must be between 0 and {self.nlevels}")
            
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
