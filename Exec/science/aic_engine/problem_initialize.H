#ifndef problem_initialize_H
#define problem_initialize_H

#include <hdf5.h>
#include <AMReX_REAL.H>
#include "aic_model_data.H"

#if BL_USE_FLOAT == 1
#define AIC_H5T H5T_NATIVE_FLOAT
#else
#define AIC_H5T H5T_NATIVE_DOUBLE
#endif

using namespace aic_model_data;

// Update function for fill values
typedef Real (*fv_updater)(const Real, const Real, const int);

AMREX_INLINE
Real get_min(const Real a, const Real b, const int n)
{
    if(a < b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

AMREX_INLINE
Real get_avg(const Real a, const Real b, const int n)
{
    return (n*a + b) / (n + 1);
}

AMREX_INLINE
void read_aic_data(const hid_t& file, const std::string& dset_name, void* buf)
{
    hid_t dset = H5Dopen1(file, dset_name.c_str());
    H5Dread(dset, AIC_H5T, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset);
}

AMREX_INLINE
void problem_initialize()
{
    for(int i = 0; i < AMREX_SPACEDIM; i++)
    {
        problem::center[i] = 0.0_rt;
    }
    
    hid_t h5file = H5Fopen(problem::model_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    
    read_aic_data(h5file, "rmin", model_x_min.begin());
    read_aic_data(h5file, "x_out", model_r_out.begin());
    read_aic_data(h5file, "z_out", model_z_out.begin());
    
    Real data_arr[NR_MODEL*NZ_MODEL];
    constexpr int numf = 4;
    std::string dset_names[numf] = {"rho", "temp", "vx", "vz"};
    int field_ids[numf] = {idens, itemp, ivelr, ivelz};
    for(int f = 0; f < numf; f++)
    {
        read_aic_data(h5file, dset_names[f].c_str(), data_arr);
        
        for(int i = 0; i < NR_MODEL; i++)
        {
            for(int j = 0; j < NZ_MODEL; j++)
            {
                model_data(field_ids[f], i, j) = data_arr[i*NZ_MODEL + j];
            }
        }
    }
    
    jfill(0) = NZ_MODEL-1;
    jfill(1) = 0;
    for(int j = 0; j < NZ_MODEL; j++) imax_fill(j) = NR_MODEL-1;
    
    fv_updater fvu = nullptr;
    int n = 0;
    if(problem::fill_underdense == 1)
    {
        fill_vals(0) = 1.0e40_rt;
        fill_vals(1) = 1.0e40_rt;
        fvu = get_min;
    }
    else if(problem::fill_underdense == 2)
    {
        fill_vals(0) = 0.0_rt;
        fill_vals(1) = 0.0_rt;
        fvu = get_avg;
    }
    else if(problem::fill_underdense > 2)
    {
        amrex::Abort("'fill_underdense' option not implemented.");
    }
    
    for(int i = 0; i < NR_MODEL; i++)
    {
        for(int j = 0; j < NZ_MODEL; j++)
        {
            const Real rho = model_data(idens, i, j);
            const Real temp = model_data(itemp, i, j); 
            
            if(rho >= problem::rho_amb)
            {
                if(j < jfill(0)) jfill(0) = j;
                if(j > jfill(1)) jfill(1) = j;
                if(i < imax_fill(j)) imax_fill(j) = i;
                fill_vals(0) = fvu(fill_vals(0), rho, n);
                fill_vals(1) = fvu(fill_vals(1), temp, n);
            }
        }
        
        n++;
    }
    
    H5Fclose(h5file);
}

#endif
