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
    fill_vals(0) = 1.0e40_rt;
    fill_vals(1) = 1.0e40_rt;
    
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
                if(rho < fill_vals(0)) fill_vals(0) = rho;
                if(temp < fill_vals(1)) fill_vals(1) = temp;
            }
        }
    }
    
    H5Fclose(h5file);
}

#endif
