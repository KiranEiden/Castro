module probdata_module

  ! Cloud parameters
  use amrex_fort_module, only : rt => amrex_real
  real(rt)        , save ::  rho_cloud, r_cloud, T_0, v_max, alpha
  
  real(rt)        , save :: xmin, xmax, ymin, ymax, zmin, zmax
      
end module probdata_module
