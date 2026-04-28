#include <aic_model_data.H>

using namespace amrex;

namespace aic_model_data
{
    // Note: We assume 2D cylindrical for now
    
    // Arrays to be copied to GPUs
    // Note that the state component comes first in the model data array
    AMREX_GPU_MANAGED Array1D<Real, 0, 1> model_x_min;
    AMREX_GPU_MANAGED Array1D<Real, 0, NR_MODEL-1> model_r_out;
    AMREX_GPU_MANAGED Array1D<Real, 0, NZ_MODEL-1> model_z_out;
    AMREX_GPU_MANAGED Array3D<Real, 0, nvars_model-1, 0, NR_MODEL-1, 0, NZ_MODEL-1> model_data;
    
    AMREX_GPU_MANAGED Array1D<int, 0, 1> jfill;
    AMREX_GPU_MANAGED Array1D<int, 0, NZ_MODEL-1> imax_fill;
    AMREX_GPU_MANAGED Array1D<Real, 0, 1> fill_vals;
}
