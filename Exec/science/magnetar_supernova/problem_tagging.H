#ifndef problem_tagging_H
#define problem_tagging_H

#include <AMReX_TagBox.H>
#include <prob_parameters.H>
#include <Castro_util.H>

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
    
    if(r < problem::refine_scale*problem::r_0 && level < problem::initial_ejecta_tagging_level)
    {
        tag(i,j,k) = TagBox::SET;
    }
}
#endif
