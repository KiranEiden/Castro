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
    constexpr Real half_pi = 1.5707963267948966_rt;
    constexpr Real four_thirds_pi = 4.1887902047863905_rt;

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

#ifdef USE_EQ_DEP
    const Real sign_y = (y < 0.0_rt) ? -1.0_rt : 1.0_rt;
    const Real x1 = x + dx/2.0_rt;
    const Real x2 = x - dx/2.0_rt;
    const Real y1 = y - sign_y*dy/2.0_rt;
    const Real y2 = y + sign_y*dy/2.0_rt;

    Real ang_dist1 = half_pi - asin(x1 / sqrt(x1*x1 + y1*y1));
    Real ang_dist2 = half_pi - asin(x2 / sqrt(x2*x2 + y2*y2)); 

    if(ang_dist1 >= eq_ang)
    {
        return;
    }
#endif

    if(r > r_dep)
    {
        if(r > depscale*r_dep)
        {
            return;
        }
        
        frac_lum *= exp(-p_dep * (r/r_dep - 1.0_rt));
    }
    
    // Set magnetar timescale in units of s
    const Real BR3sa = B_mag * R_mag*R_mag*R_mag * sin_alph;
    const Real t_m = 2047.49866274_rt * I_mag * P_0*P_0 / (4.0_rt*BR3sa*BR3sa);
    
    // Magnetar initial rotational kinetic energy, units of erg
    const Real E_0 = 1.97392088e+52_rt * I_mag / (P_0*P_0);
    
    // Evaluate at start of timestep
    const Real tfac = time/t_m + 1.0_rt;
    const Real L_tot = E_0 * (k_eng - 1.0_rt) / (t_m * pow(tfac, k_eng));

    // Estimate drhoe_dt, where energy is deposited uniformly over the volume
    const Real V_dep = four_thirds_pi * r_dep*r_dep*r_dep;
    Real drhoe_dt = L_tot / V_dep * frac_lum;
#ifdef USE_SIN_DEP
    const Real sin_th = x/r;
    drhoe_dt *= 1.5_rt * sin_th*sin_th;
#endif
#ifdef USE_COS_DEP
    const Real cos_th = y/r;
    drhoe_dt *= 3.0_rt * cos_th*cos_th;
#endif    
#ifdef USE_EQ_DEP
    const Real tan_ph = tan(eq_ang);
    Real vol_rat = 1.0_rt;
    if(ang_dist2 > eq_ang)
    {
        const Real x_lo = max(x2, y1 * sign_y / tan_ph);
        const Real x_hi = min(x1, y2 * sign_y / tan_ph);
        vol_rat = 2.0_rt/3.0_rt * (x_hi*x_hi*x_hi - x_lo*x_lo*x_lo) * tan_ph
             - y1 * (x_hi*x_hi - x_lo*x_lo) * sign_y
             + dy * (x1*x1 - x_hi*x_hi);
        vol_rat /= 2.0_rt * x * dx * dy;
    }

    drhoe_dt *= vol_rat / sin(eq_ang);
#endif
    
    src(i,j,k,UEINT) += drhoe_dt;
    src(i,j,k,UEDEN) += drhoe_dt;
#ifdef USE_MASS_SOURCE
    const Real drho_dt = mass_mult * (4.0_rt/3.0_rt - 1.0_rt) * drhoe_dt / (c*c);
    src(i,j,k,URHO)  += drho_dt;
    
    // Calculate magnetar mass
    const Real M_mag = 2.5e+33_rt*I_mag/(R_mag*R_mag);
    // Deposit the mass with an initial velocity equal to the escape velocity
    const Real v = sqrt(4.0_rt*G*M_mag/dr);
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
