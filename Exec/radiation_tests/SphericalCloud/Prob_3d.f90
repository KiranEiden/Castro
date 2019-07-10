subroutine amrex_probinit(init, name, namlen, problo, probhi) bind(C, name="amrex_probinit")

  use probdata_module
  use network, only: nspec
  use amrex_error_module
  use eos_module
  use eos_type_module, only: eos_t, eos_input_rt
  use amrex_fort_module, only: rt => amrex_real
  implicit none
  
  integer, intent(in) :: init, namlen
  integer, intent(in) :: name(namlen)
  double precision, intent(in) :: problo(:), probhi(:)
  
  type(eos_t) :: eos_state
  integer :: untin, i
  
  namelist /fortin/ r_cloud, rho_cloud, T_0, v_max, alpha
  
  !     Build "probin" filename -- the name of file containing fortin namelist.
  !
  integer, parameter :: maxlen = 256
  character probin*(maxlen)
  
  if(namlen .gt. maxlen) then
    call amrex_error('probin file name too long')
  end if
  
  do i = 1, namlen
     probin(i:i) = char(name(i))
  end do
  
  ! set defaults
  r_cloud = 3.086d17
  rho_cloud = 1.6156913825823182d-18
  T_0 = 1.d5
  v_max = 0.d0
  alpha = 1.d-3

  ! Read probin file
  untin = 9
  open(untin,file=probin(1:namlen),form='formatted',status='old')
  read(untin,fortin)
  close(unit=untin)
  
  eos_state % rho = rho_cloud
  eos_state % T   = T_0
  eos_state % xn  = 0.e0_rt
  eos_state % xn(1) = 1.e0_rt
  
  call eos(eos_input_rt, eos_state)

  rhoe_cloud = rho_cloud * eos_state % e

  eos_state % rho = alpha * rho_cloud
  eos_state % T   = alpha * T_0
  eos_state % xn  = 0.e0_rt
  eos_state % xn(1) = 1.e0_rt

  call eos(eos_input_rt, eos_state)

  rhoe_ext = alpha * rho_cloud * eos_state % e

end subroutine amrex_probinit

! ::: -----------------------------------------------------------
! ::: This routine is called at problem setup time and is used
! ::: to initialize data on each grid.
! :::
! ::: NOTE:  all arrays have one cell of ghost zones surrounding
! :::        the grid interior.  Values in these cells need not
! :::        be set here.
! :::
! ::: INPUTS/OUTPUTS:
! :::
! ::: level     => amr level of grid
! ::: time      => time at which to init data
! ::: lo,hi     => index limits of grid interior (cell centered)
! ::: nstate    => number of state components.  You should know
! :::		   this already!
! ::: state     <=  Scalar array
! ::: delta     => cell size
! ::: xlo,xhi   => physical locations of lower left and upper
! :::              right hand corner of grid.  (does not include
! :::		   ghost region).
! ::: -----------------------------------------------------------
subroutine ca_initdata(level, time, lo, hi, nscal, &
                       state, state_lo, state_hi, &
                       delta, plo, phi)

  use probdata_module
  use meth_params_module, only: NVAR, URHO, UMX, UMY, UMZ, UEDEN, UEINT, UFS, UFX, UTEMP
  use network, only: nspec, naux

  use amrex_fort_module, only : rt => amrex_real

  implicit none

  integer, intent(in) :: level, nscal
  integer, intent(in) :: lo(3), hi(3)
  integer, intent(in) :: state_lo(3), state_hi(3)
  real(rt), intent(in) :: plo(3), phi(3), time, delta(3)
  real(rt), intent(inout) :: state(state_lo(1):state_hi(1), &
                                   state_lo(2):state_hi(2), &
                                   state_lo(3):state_hi(3), nscal)

  real(rt) :: x, y, z, r, fac, rhoe
  integer :: i, j, k

  do k = lo(3), hi(3)
      
    z = plo(3) + delta(3)*(float(k) + 0.5e0_rt)
    
    do j = lo(2), hi(2)
        
        y = plo(2) + delta(2)*(float(j) + 0.5e0_rt)
        
        do i = lo(1), hi(1)
            
           x = plo(1) + delta(1)*(float(i) + 0.5e0_rt)
           r = sqrt(x**2 + y**2 + z**2)
           
           if(r > r_cloud) then
              fac = alpha
              rhoe = rhoe_ext
           else
              fac = 1.e0_rt
              rhoe = rhoe_cloud
           end if
           
           state(i, j, k, URHO) = fac * rho_cloud
           state(i, j, k, UMX) = fac * v_max * x * rho_cloud / r_cloud
           state(i, j, k, UMY) = fac * v_max * y * rho_cloud / r_cloud
           state(i, j, k, UMZ) = fac * v_max * z * rho_cloud / r_cloud
           state(i, j, k, UEDEN) = rhoe
           state(i, j, k, UEINT) = rhoe
           state(i, j, k, UTEMP) = fac * T_0

           ! set the composition to be all in the first species
           state(i, j, k, UFS:UFS+nspec-1) = state(i, j, k, URHO) / nspec
           ! state(i, j, k, UFS) = state(i, j, k, URHO)
           if (naux > 0) then
              state(i, j, k, UFX) = state(i, j, k, URHO)
           end if

        end do
     end do
  end do

end subroutine ca_initdata
