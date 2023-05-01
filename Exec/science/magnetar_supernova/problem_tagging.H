#ifndef problem_tagging_H
#define problem_tagging_H

#include <AMReX_TagBox.H>
#include <prob_parameters.H>
#include <Castro_util.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
const Real get_magvel(int i, int j, int k, Array4<Real const> const& state)
{
    const Real vx = state(i,j,k,UMX) / state(i,j,k,URHO);
        
#if AMREX_SPACEDIM > 1
    const Real vy = state(i,j,k,UMY) / state(i,j,k,URHO);
#else
    const Real vy = 0.0_rt;
#endif
#if AMREX_SPACEDIM > 2
    const Real vz = state(i,j,k,UMZ) / state(i,j,k,URHO);
#else
    const Real vz = 0.0_rt;
#endif
    
    return sqrt(vx*vx + vy*vy + vz*vz);
}

///
/// Define problem-specific tagging criteria
///
/// @param i         x-index
/// @param j         y-index
/// @param k         z-index
/// @param tag       tag array (TagBox)
/// @param state     simulation state (Fab)
/// @param level     AMR level
/// @param geomdata  geometry data
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_tagging(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level, const GeometryData& geomdata)
{
    const Real dx = geomdata.CellSize(0);
    const Real x = geomdata.ProbLo(0) + (static_cast<Real>(i) + 0.5_rt)*dx;
#if AMREX_SPACEDIM > 1
    const Real dy = geomdata.CellSize(1);
    const Real y = geomdata.ProbLo(1) + (static_cast<Real>(j) + 0.5_rt)*dy;
#else
    const Real dy = 0.0_rt;
    const Real y = 0.0_rt;
#endif
#if AMREX_SPACEDIM > 2
    const Real dz = geomdata.CellSize(2);
    const Real z = geomdata.ProbLo(2) + (static_cast<Real>(k) + 0.5_rt)*dz;
#else
    const Real dz = 0.0_rt;
    const Real z = 0.0_rt;
#endif

    const Real r = sqrt(x*x + y*y + z*z);
    const Real rr_ej = problem::refine_scale_ej*problem::r_0; 
    const Real rr_bub = problem::f_dep*problem::r_0 * problem::refine_scale_bub;

    if(r < rr_bub && level < problem::bubble_tagging_level)
    {
	// Initial refinement region to encapsulate bubble
        tag(i,j,k) = TagBox::SET;
    }
    else if(r < rr_ej && level < problem::ejecta_tagging_level)
    {
	// Initial refinement region to encapsulate ejecta
        tag(i,j,k) = TagBox::SET;
    }
    
    // Calculate density gradients
    Real grad_x = 0.0_rt;
    Real grad_y = 0.0_rt;
    Real grad_z = 0.0_rt;
    bool grad_flag = false;
    
    const Real dens = state(i,j,k,URHO);
    
    if(i > 0)
    {
        grad_x = 1.0_rt - state(i-1,j,k,URHO) / dens;
    }
    
#if AMREX_SPACEDIM > 1
    if(j > 0)
    {
        grad_y = 1.0_rt - state(i,j-1,k,URHO) / dens;
    }
#endif
#if AMREX_SPACEDIM > 2
    if(k > 0)
    {
        grad_z = 1.0_rt - state(i,j,k-1,URHO) / dens;
    }
#endif

    grad_flag = grad_x <= problem::dens_grad_crit || grad_y <= problem::dens_grad_crit ||
            grad_z <= problem::dens_grad_crit;
    
    if(grad_flag)
    {
        if(dens < problem::dens_crit && level < problem::ejecta_tagging_level)
        {
            tag(i,j,k) = TagBox::SET;
        }
        else if(dens >= problem::dens_crit && level < problem::bubble_tagging_level)
        {
            tag(i,j,k) = TagBox::SET;
        }
    }
}
#endif
