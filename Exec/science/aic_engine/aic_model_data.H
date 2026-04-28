#ifndef aic_model_data_H
#define aic_model_data_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuContainers.H>

#include <network_properties.H>

#define NR_MODEL 80
#define NZ_MODEL 160

using namespace amrex;

namespace aic_model_data
{
    // Note: We assume 2D cylindrical for now
    
    // Number of state components + indices for each
    constexpr int nvars_model = 5 + NumSpec + NumAux;
    constexpr int idens = 0;
    constexpr int itemp = 1;
    constexpr int ivelr = 2;
    constexpr int ivelz = 3;
    constexpr int ispec = 4;
    constexpr int iaux = ispec + NumSpec;
    
    // Arrays to be copied to GPUs
    // Note that the state component comes first in the model data array
    extern AMREX_GPU_MANAGED Array1D<Real, 0, 1> model_x_min;
    extern AMREX_GPU_MANAGED Array1D<Real, 0, NR_MODEL-1> model_r_out;
    extern AMREX_GPU_MANAGED Array1D<Real, 0, NZ_MODEL-1> model_z_out;
    extern AMREX_GPU_MANAGED Array3D<Real, 0, nvars_model-1, 0, NR_MODEL-1, 0, NZ_MODEL-1> model_data;
    
    // Used for filling underdense regions
    extern AMREX_GPU_MANAGED Array1D<int, 0, 1> jfill;
    extern AMREX_GPU_MANAGED Array1D<int, 0, NZ_MODEL-1> imax_fill;
    extern AMREX_GPU_MANAGED Array1D<Real, 0, 1> fill_vals;
    
    AMREX_INLINE AMREX_GPU_HOST_DEVICE
    void interpolate_reg_grid(const Real r, const Real z,
                              const int* field_arr, Real* out,
                              const int numf)
    {
        const Real dr = model_r_out(0) - model_x_min(0);
        const Real dz = model_z_out(0) - model_x_min(1);
        
        int i0 = static_cast<int>((r - model_x_min(0)) / dr);
        int j0 = static_cast<int>((z - model_x_min(1)) / dz);
        if(i0 < 0) i0 = 0;
        if(j0 < 0) j0 = 0;
        if(i0 > NR_MODEL-1) i0 = NR_MODEL-1;
        if(j0 > NZ_MODEL-1) j0 = NZ_MODEL-1;
        
        const Real r0 = model_r_out(i0) - 0.5_rt*dr;
        const Real z0 = model_z_out(j0) - 0.5_rt*dz;
        
        int i1 = (r > r0) ? i0 + 1 : i0 - 1;
        int j1 = (z > z0) ? j0 + 1 : j0 - 1;
        if(i1 < 0) i1 = 1;
        if(j1 < 0) j1 = 1;
        if(i1 > NR_MODEL-1) i1 = NR_MODEL-2;
        if(j1 > NZ_MODEL-1) j1 = NZ_MODEL-2;
        
        const Real r1 = model_r_out(i1) - 0.5_rt*dr;
        const Real z1 = model_z_out(j1) - 0.5_rt*dz;
        
        const Real rwgt = (r - r0) / (r1 - r0); 
        const Real zwgt = (z - z0) / (z1 - z0); 
        
        for(int ifield = 0; ifield < numf; ifield++)
        {
            out[ifield] = model_data(field_arr[ifield], i0, j0) * (1 - rwgt) * (1 - zwgt)
                        + model_data(field_arr[ifield], i1, j0) * rwgt * (1 - zwgt)
                        + model_data(field_arr[ifield], i0, j1) * (1 - rwgt) * zwgt
                        + model_data(field_arr[ifield], i1, j1) * rwgt * zwgt;
        }
    }
}

#endif
