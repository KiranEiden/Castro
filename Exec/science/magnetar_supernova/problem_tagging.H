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
/// @param time      simulation time
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_tagging(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level, const GeometryData& geomdata,
		     const Real time)
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
    const Real rr_ej = problem::refine_scale_ej*problem::r_0 + time*problem::ej_tag_vel; 
    const Real rr_bub = problem::f_dep*problem::r_0*problem::depscale * problem::refine_scale_bub + time*problem::bub_tag_vel;

    /*    
    if(state(i,j,k,URHO) < problem::csm_derefine_dens && level == 0)
    {
        tag(i,j,k) = TagBox::CLEAR;
    }
    else if(state(i,j,k,URHO) < problem::ej_derefine_dens && level >= problem::ejecta_tagging_level)
    {
        tag(i,j,k) = TagBox::CLEAR;
    }
    */

    /*
    if(state(i,j,k,QPRES) < problem::pressure_crit && level < problem::bubble_tagging_level)
    {
        tag(i,j,k) = TagBox::SET;
    }
    */

    if(r < rr_bub && level < problem::bubble_tagging_level)
    {
	// Refinement region to encapsulate bubble
        tag(i,j,k) = TagBox::SET;
    }
    else if(r < rr_ej && level < problem::ejecta_tagging_level)
    {
	// Refinement region to encapsulate ejecta
        tag(i,j,k) = TagBox::SET;
    }
}
#endif
