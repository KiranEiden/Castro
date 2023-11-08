#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <eos.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    constexpr Real c = 2.99792458e+10_rt;
    constexpr Real G = 6.67430e-8_rt;
    constexpr Real k_b = 1.3807e-16_rt;
    constexpr Real M_sun = 1.98847e33_rt;

    Real frac_lum = frac_lum_base;
    if(time < t_mag_act)
    {
        frac_lum *= time / t_mag_act;
    }
    
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

    // Radial cell width approximation
    const Real dr = sqrt(dx*dx + dy*dy + dz*dz);
    // Radius associated with the midpoint of the cell
    const Real r = sqrt(x*x + y*y + z*z);
    // Approximate radial cell boundaries
    const Real r_hi = r + 0.5_rt*dr;
    const Real r_lo = r - 0.5_rt*dr;
    
    // "Deposition radius" to set initial energy deposition profile
    const Real r_dep = f_dep * (r_0 + beta*c*time);
    //const Real r_dep = f_dep * r_0;
    
    if(r > r_dep)
    {
        if(r > depscale*r_dep)
        {
            return;
        }
        
        frac_lum *= exp(-p_dep * (r/r_dep - 1.0_rt));
    }
    
    // Ejecta and CSM mass
    // const Real M_ej = m_0 * 1.989e33_rt;
    // const Real M_csm = M_ej * eta;
    
    // Transition radius
    const Real R_t = f_r * r_0;
    
    // Ejecta density
    // const Real rho0 = M_ej / (4.1887902047863905_rt * r_0*r_0*r_0);
    
    // Opacity estimate, set so r_dep is at roughly \tau = 1
    // const Real kappa = 1.0 / (r_dep * rho0);
    
    // Set magnetar timescale in units of s
    const Real BR3sa = B_mag * R_mag*R_mag*R_mag * sin_alph;
    const Real t_m = 2047.49866274_rt * I_mag * P_0*P_0 / (4.0*BR3sa*BR3sa);
    
    // Magnetar initial rotational kinetic energy, units of erg
    const Real E_0 = 1.97392088e+52_rt * I_mag / (P_0*P_0);
    
    // Evaluate at start of timestep
    const Real tfac = time/t_m + 1.0_rt;
    const Real L_tot = E_0 / (t_m * tfac * tfac);

	// Estimate drhoe_dt (rate of change of internal energy density)
// #if AMREX_SPACEDIM == 1
    // const Real V_cell = 12.5663706143592_rt*r*r*dr + 1.04719755119660_rt*dr*dr*dr;
    // Real tau = 0.0;
    // for(int ii = 0; ii < i; ii++)
    // {
        // tau += state(ii,j,k,URHO) * kappa * dr;
    // }
    // const Real tau_zone = state(i,j,k,URHO) * kappa * dr;
	// const Real frac_abs = exp(-tau) * (1.0_rt - exp(-tau_zone));
	// const Real drhoe_dt = frac_abs * L_tot / V_cell;
// #else
	// amrex::Abort('Magnetar source term not implemented for multi-D.');
// #endif

    // Estimate drhoe_dt, where energy is deposited uniformly over the volume
    const Real V_dep = 4.1887902047863905_rt * r_dep*r_dep*r_dep;
#ifdef USE_ASYM_DEP
    const Real sin_th = x/r;
    const Real drhoe_dt = L_tot / V_dep * frac_lum * 1.5_rt * sin_th*sin_th;
#else
    const Real drhoe_dt = L_tot / V_dep * frac_lum;
#endif

    // // Estimate drhoe_dt with parabolic energy deposition dropoff
    // const Real V_cell = 12.5663706143592_rt*r*r*dr + 1.04719755119660_rt*dr*dr*dr;
    // const Real r_hi_tild = (r_hi > r_dep) ? r_dep : r_hi;
    // const Real dr_tild = r_hi_tild - r_lo;
    // const Real frac_abs = 3.0_rt / (2.0_rt * r_dep) * (dr_tild - (3.0_rt*r_lo*r_lo*dr_tild
            // + 3.0_rt*r_lo*dr_tild*dr_tild + dr_tild*dr_tild*dr_tild) / (3.0_rt*r_dep*r_dep));
    // const Real drhoe_dt = frac_abs * L_tot / V_cell;
    
    src(i,j,k,UEINT) += drhoe_dt;
    src(i,j,k,UEDEN) += drhoe_dt;
#ifdef USE_MASS_SOURCE
    const Real drho_dt = mass_mult * (eos_gamma - 1.0_rt) * drhoe_dt / (c*c);
    src(i,j,k,URHO)  += drho_dt;
    
    // Deposit the mass with an initial velocity equal to the escape velocity
    const Real v = sqrt(2.0_rt*G*m_0*M_sun/dr);
#if AMREX_SPACEDIM == 1
    const Real v_x = v;
    const Real v_y = 0.0_rt;
    const Real v_z = 0.0_rt;
#elif AMREX_SPACEDIM == 2
    const Real v_x = v * x/r;
    const Real v_y = v * y/r;
    const Real v_z = 0.0_rt;
#else
    const Real v_x = v * x/r;
    const Real v_y = v * y/r;
    const Real v_z = v * z/r;
#endif
    src(i,j,k,UMX) += drho_dt * v_x;
    src(i,j,k,UMY) += drho_dt * v_y;
    src(i,j,k,UMZ) += drho_dt * v_z;
    src(i,j,k,UEDEN) += 0.5_rt * drho_dt * v*v;

    // Inject mass as last species
    src(i,j,k,UFS+NumSpec-1) += drho_dt;
#endif
}

#endif
