module probdata_module

  ! Cloud parameters
  use amrex_fort_module, only : rt => amrex_real
  real(rt), allocatable ::  rho_cloud, r_cloud, T_0, v_max, alpha, rhoe_cloud, rhoe_ext
  
  real(rt), allocatable :: xmin, xmax, ymin, ymax, zmin, zmax
      
end module probdata_module
