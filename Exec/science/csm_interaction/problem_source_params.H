#ifndef problem_source_params_H
#define problem_source_params_H

#include <prob_parameters.H>

struct SrcParams
{
public:

    SrcParams()
    {
        // "Deposition radius" to set initial energy deposition profile
        const Real r_dep = problem::f_dep * problem::r_0;
        
        // Ejecta and CSM mass
        const Real M_ej = problem::m_0 * Msun;
        const Real M_csm = M_ej * problem::eta;
        
        // Transition radius
        const Real R_t = problem::f_r * problem::r_0;
        
        // Ejecta density
        const Real rho0_ej = (problem::n-3.0_rt)*(3.0_rt-problem::d) /
                (4.0_rt*M_PI*(problem::n-problem::d)) * M_ej/pow(R_t,3);
        
        // Opacity estimate, set so r_dep is at roughly \tau = 1
        const Real kappa = 1.0 / (r_dep * rho0_ej);
        
        // Set magnetar timescale in units of s
        const Real BR3sa = problem::B_mag * pow(problem::R_mag, 3) * problem::sin_alph;
        const Real t_m = 2047.49866274_rt * problem::I_mag * pow(problem::P_0, 2) / (4.0*pow(BR3sa, 2));
        
        // Magnetar initial rotational kinetic energy, units of erg
        const Real E_0 = 1.97392088e+52_rt * problem::I_mag / pow(problem::P_0, 2);
    }
    
    // Physical constants
    constexpr Real Msun = 1.989e33_rt;
    constexpr Real clight = 2.998e10_rt;
    constexpr Real km_to_cm = 1.e5_rt;
    
    // "Deposition radius" to set initial energy deposition profile
    const Real r_dep;
    
    // Ejecta and CSM mass
    const Real M_ej;
    const Real M_csm;
    
    // Transition radius
    const Real R_t;
    
    // Ejecta density
    const Real rho0_ej;
    
    // Opacity estimate, set so r_dep is at roughly \tau = 1
    const Real kappa = 1.0 / (r_dep * rho0_ej);
    
    // Set magnetar timescale in units of s
    const Real BR3sa;
    const Real t_m;
    
    // Magnetar initial rotational kinetic energy, units of erg
    const Real E_0;
};

#endif
