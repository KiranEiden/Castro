#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <aic_model_data.H>
#include <eos.H>
#include <AMReX_Random.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real model_index_to_min_coord(const int idx, const int dir)
{
    if(dir == 0)
    {
        if(idx == 0)
        {
            return aic_model_data::model_x_min(0);
        }
        else
        {
            return aic_model_data::model_r_out(idx-1);
        }
    }
    else if(dir == 1)
    {
        if(idx == 0)
        {
            return aic_model_data::model_x_min(1);
        }
        else
        {
            return aic_model_data::model_z_out(idx-1);
        }
    }
    else
    {
        amrex::Abort("Invalid direction passed to `model_index_to_min_coord` function.");
        return 0.0_rt;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
int coord_to_model_index(const Real coord, const int dir)
{
    if(dir == 0)
    {
        const Real drm = aic_model_data::model_r_out(0) - aic_model_data::model_x_min(0);
        return static_cast<int>((coord - aic_model_data::model_x_min(0)) / drm);
    }
    else if(dir == 1)
    {
        const Real dzm = aic_model_data::model_z_out(0) - aic_model_data::model_x_min(1);
        return static_cast<int>((coord - aic_model_data::model_x_min(1)) / dzm);
    }
    else
    {
        amrex::Abort("Invalid direction passed to `coord_to_model_index` function.");
        return -1;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k,
                                    Array4<Real> const& state,
                                    const GeometryData& geomdata)
{
    const Real* dx = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();
    const Real* probhi = geomdata.ProbHi();
    
    Real r = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5_rt) - problem::center[0];
    Real z = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5_rt) - problem::center[1];
    
    const Real model_r_max = aic_model_data::model_r_out(NR_MODEL-1);
    const Real model_z_max = aic_model_data::model_z_out(NZ_MODEL-1);
    const bool model_cell = (r <= model_r_max) && (z <= model_z_max) && (z >= -model_z_max);
    
    Real rho, temp, velr, velz;
    rho = temp = velr = velz = 0.0_rt;
    
    constexpr int numf = aic_model_data::nvars_model;
    int field_arr[numf];
    Real interp_state[numf];
        
    if(model_cell)
    {
        for(int ifield = 0; ifield < numf; ifield++)
        {
            field_arr[ifield] = ifield;
        }

        aic_model_data::interpolate_reg_grid(r, z, field_arr, interp_state, numf);
        
        rho = interp_state[aic_model_data::idens];
        temp = interp_state[aic_model_data::itemp];
        velr = interp_state[aic_model_data::ivelr];
        velz = interp_state[aic_model_data::ivelz];
    }
    
    const bool amb_cell = !model_cell || (rho <= problem::rho_amb) || (temp <= problem::T_amb);
    bool fill_cell = false;
    
    if(problem::fill_underdense)
    {
        const Real zfill_min = model_index_to_min_coord(aic_model_data::jfill(0), 1);
        const Real zfill_max = model_index_to_min_coord(aic_model_data::jfill(1), 1);
        const int jmodel = coord_to_model_index(z, 1);
        Real rfill_max;
        if(jmodel < aic_model_data::jfill(0) || jmodel > aic_model_data::jfill(1))
        {
            rfill_max = problo[0];
        }
        else
        {
            rfill_max = model_index_to_min_coord(aic_model_data::imax_fill(jmodel), 0);
        }
        
        fill_cell = amb_cell && (z > zfill_min) && (z < zfill_max) && (r < rfill_max);
    }
    
    if(amb_cell)
    {
        if(fill_cell)
        {
            rho = aic_model_data::fill_vals(0);
            temp = aic_model_data::fill_vals(1);
            velr = velz = 0.0_rt;
        }
        else
        {
            rho = problem::rho_amb;
            temp = problem::T_amb;
            velr = velz = 0.0_rt;
        }
    }
    
    state(i, j, k, URHO) = rho;
    state(i, j, k, UTEMP) = temp;
    state(i, j, k, UMX) = velr*rho;
    state(i, j, k, UMY) = velz*rho;
    state(i, j, k, UMZ) = 0.0_rt;
    
    for(int ispec = 0; ispec < NumSpec; ispec++)
    {
        state(i, j, k, UFS+ispec) = 0.0_rt;
    }
    
    if(amb_cell)
    {
        state(i, j, k, UFS) = rho;

        if(NumAux > 0)
        {
            state(i, j, k, UFX + AuxZero::iabar) = rho*aion[0];
        }
        if(NumAux > 1)
        {
            state(i, j, k, UFX + AuxZero::iye) = rho*(zion[0] / aion[0]);
        }
    }
    else
    {
        Real* interp_spec = &interp_state[aic_model_data::ispec];
        Real* interp_aux = &interp_state[aic_model_data::iaux];
        for(int ispec = 0; ispec < NumSpec; ispec++)
        {
            state(i, j, k, UFS+ispec) = rho * interp_spec[ispec];
        }
        for(int iaux = 0; iaux < NumAux; iaux++)
        {
            state(i, j, k, UFX+iaux) = rho * interp_aux[iaux];
        }
    }
    
    eos_t eos_state;
    eos_state.rho = state(i, j, k, URHO);
    eos_state.T = state(i, j, k, UTEMP);
    for(int ispec = 0; ispec < NumSpec; ispec++)
    {
        eos_state.xn[ispec] = state(i, j, k, UFS+ispec) / state(i, j, k, URHO);
    }
    for(int iaux = 0; iaux < NumAux; iaux++)
    {
        eos_state.aux[iaux] = state(i, j, k, UFX+iaux) / state(i, j, k, URHO);
    }
    eos(eos_input_rt, eos_state);
    
    state(i, j, k, UEINT) = state(i, j, k, URHO) * eos_state.e;
    state(i, j, k, UEDEN) = state(i, j, k, UEINT) + 0.5_rt *
            (state(i, j, k, UMX) * state(i, j, k, UMX) +
             state(i, j, k, UMY) * state(i, j, k, UMY) +
             state(i, j, k, UMZ) * state(i, j, k, UMZ)) / state(i, j, k, URHO);
}

#endif
