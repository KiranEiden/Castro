#ifndef problem_initialize_rad_data_H
#define problem_initialize_rad_data_H

#include <fundamental_constants.H>
#include <prob_parameters.H>
#include <blackbody.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_rad_data (int i, int j, int k,
								  Array4<Real> const& rad,
								  GpuArray<Real, NGROUPS+1> const& xnu,
								  GpuArray<Real, NGROUPS> const& nugroup,
								  GpuArray<Real, NGROUPS> const& dnugroup,
								  const GeometryData& geomdata)
{

  const Real* dx = geomdata.CellSize();
  const Real* problo = geomdata.ProbLo();
  const Real* probhi = geomdata.ProbHi();

  Real R_ej = problem::r_0;
  Real R_csm = (1.0_rt + problem::delt + problem::xi_bo)*problem::r_0;


  Real x = problo[0] + (static_cast<Real>(i)+0.5_rt)*dx[0];

  Real y = 0.0;
#if AMREX_SPACEDIM >= 2
  y = problo[1] + (static_cast<Real>(j)+0.5_rt)*dx[1];
#endif
  Real z = 0.0;
#if AMREX_SPACEDIM == 3
  z = problo[2] + (static_cast<Real>(k)+0.5_rt)*dx[2];
#endif

  Real r = std::sqrt(x*x + y*y + z*z);

  Real T = 0.0_rt;

  if(r < R_ej) T = problem::T_ej;
  else T = problem::T_0;

  if(NGROUPS == 1){
    rad(i,j,k,0) = C::a_rad*pow(T,4);
  }
  else
  {
    for(int g = 0; g < NGROUPS; ++g)
    {
      rad(i,j,k,g) = BGroup(T,xnu[g],xnu[g+1]);
    }
  }


}
#endif
