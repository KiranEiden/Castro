#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k,
                                    Array4<Real> const& state,
                                    const GeometryData& geomdata)
{

  const Real* dx = geomdata.CellSize();
  const Real* problo = geomdata.ProbLo();
  const Real* probhi = geomdata.ProbHi();

  constexpr Real Msun = 1.989e33_rt;
  constexpr Real clight = 2.998e10_rt;
  constexpr Real km_to_cm = 1.e5_rt;

  Real n = problem::n;
  Real d = problem::d;
  Real s = problem::s;

  Real R_t  = problem::f_r*problem::r_0;
  Real M_ej = problem::m_0*Msun;
  Real M_csm = M_ej*problem::eta;
  Real R_csm = problem::r_0;
  Real R_w = problem::R_w;

  Real xi = problem::xi_bo;

  Real v_t = problem::beta*clight;

//  Real  tau_0 = problem::kap*M_csm/R_csm/R_csm;
  Real mpi = 3.14159e0_rt;
//  BL_ASSERT(problem::tau < tau_0/(4.e0_rt*mpi));

  Real delt = problem::delt;//0.5*(sqrt(3.e0_rt*(tau_0/problem::tau)/mpi-3.0e0_rt)-3.0e0_rt);

  Real R_out = (1.0_rt+delt)*problem::r_0;
	
  //Real v_t = problem::r_0/problem::t_0;
  
  Real rho_0 = 5.01410457e19_rt * problem::Mdot / problem::v_w / (R_csm*R_csm);
  Real rho_a = problem::f_a*rho_0;


  Real x = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5_rt) - problem::center[0];

  Real y = 0.0;
#if AMREX_SPACEDIM >= 2
  y = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5_rt) - problem::center[1];
#endif

  Real z = 0.0;
#if AMREX_SPACEDIM == 3
  z = problo[2] + dx[2] * (static_cast<Real>(k) + 0.5_rt) - problem::center[2];
#endif

  Real r = std::sqrt(x*x + y*y + z*z);

  Real rho = 0.0_rt;
  Real T = 0.0_rt;
  Real v = 0.0_rt;

  Real rho0_ej;
  if(problem::pl_ej > 0)
  {
      Real c_rho = 1.0_rt / (3.0_rt - d) + (1.0_rt - pow(problem::f_r, n-3)) / (n - 3.0_rt);
      rho0_ej = M_ej / (4.0_rt*M_PI*pow(R_t,3)) / c_rho;
  }
  else
  {
      rho0_ej = 3.0_rt/(4.0_rt*M_PI)*M_ej/pow(R_csm,3);
  }

  Real rho0_csm;
  if(problem::f_r < 1.0_rt)
  {
    rho0_csm = rho0_ej * pow(problem::f_r, n);
  }
  else
  {
    rho0_csm = rho0_ej * pow(problem::f_r, d);
  }
  Real p = -log(rho0_csm/rho_a) / log(R_csm/R_out);
  
  //Real rho0_csm = (3.0_rt - s)/(4.0_rt*M_PI)
           //(pow(1.0_rt + delt,3-s)-1.0_rt)
           // * M_csm/pow(R_csm,3);
  //Real rho0_bo = rho0_csm*pow(R_out/R_csm,-s);

  // ejecta portion
  if(r < R_csm)
  {
    rho = rho0_ej;

    if(problem::pl_ej > 0)
    {
      if( r < R_t) rho *= pow(r/R_t,-d);
      else rho *= pow(r/R_t,-n);

      T = problem::T_ej;
      v = v_t*(r/R_t);
    }
	state(i,j,k,UFX) = 0.0;
  }
  else // CSM/ambient portion
  {
    if(r < R_out)
    {
      rho = rho0_csm * pow(r/R_csm,-p);
    }
    else
    {
      rho = rho_a * pow(r/R_out,-s);
    }
    
    //if(r < R_out)
    //{
      //rho = rho0_csm*pow(r/R_csm,-s);
      //state(i,j,k,UFX) = rho;
    //}
    //else
    //{
      //rho = rho0_csm*pow(1.+delt,-s)*pow(r/R_out,-problem::p);
      //state(i,j,k,UFX) = 0.0;
    //}
    //else
    //{
      //rho = rho0_csm*pow(1.0_rt+problem::delt,-s)*exp(-0.5e0_rt*pow(r-R_out,2)/pow(xi*problem::delt*R_csm,2));
    //}
     
    rho = max(rho,1.01*small_dens);
    T = problem::T_0;
    v = problem::v_csm*km_to_cm;
  }
  
  state(i,j,k,URHO) = rho;
  state(i,j,k,UTEMP) = T;

  // need to convert into 3d vector
  Real v_x = 0.0_rt;
  Real v_y = 0.0_rt;
  Real v_z = 0.0_rt;
#if AMREX_SPACEDIM == 1 // spherical
  v_x = v;
#elif AMREX_SPACEDIM == 2 // cylindrical
  Real phi = std::atan(y/x);
  v_x = v*std::cos(phi);
  v_y = v*std::sin(phi);
#elif AMREX_SPACEDIM == 3 // cartesian
  Real theta = std::acos(z/r);
  Real phi   = std::atan(y/x);
  v_x = v*std::cos(phi)*std::sin(theta);
  v_y = v*std::sin(phi)*std::sin(theta);
  v_z = v*std::cos(theta);
#endif
  state(i,j,k,UMX) = rho*v_x;
  state(i,j,k,UMY) = rho*v_y;
  state(i,j,k,UMZ) = rho*v_z;

  state(i,j,k,UFS) = rho;
//state(i,j,k,UFX) = 0.0;

  eos_t eos_state;
  eos_state.rho = rho;
  eos_state.T = T;
  for(int n = 0; n < NumSpec; ++n) {
    eos_state.xn[n] = state(i,j,k,UFS+n) / state(i,j,k,URHO);
  }
  eos(eos_input_rt,eos_state);

  state(i,j,k,UEINT) = state(i,j,k,URHO) * eos_state.e;
  state(i,j,k,UEDEN) = state(i,j,k,UEINT) + 0.5_rt *
              (state(i,j,k,UMX)*state(i,j,k,UMX) +
               state(i,j,k,UMY)*state(i,j,k,UMY) +
               state(i,j,k,UMZ)*state(i,j,k,UMZ)) / state(i,j,k,URHO);

  state(i,j,k,UTEMP) = T;
}

#endif
