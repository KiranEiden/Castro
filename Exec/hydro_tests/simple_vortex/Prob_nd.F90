! This sets up the Gresho vortex problem as described in 
! Miczek, Roeple, and Edelmann 2015
!
! By choosing the reference pressure, p0, we can specify the
! Mach number

subroutine amrex_probinit (init, name, namlen, problo, probhi) bind(c)

  use probdata_module
  use eos_module
  use eos_type_module, only: eos_t, eos_input_rt
  use castro_error_module
  use amrex_fort_module, only: rt => amrex_real

  implicit none

  integer,  intent(in) :: init, namlen
  integer,  intent(in) :: name(namlen)
  real(rt), intent(in) :: problo(:), probhi(:)

  type(eos_t) :: eos_state
  integer :: untin, i

  namelist /fortin/ rho, temp, vel

  ! Build "probin" filename -- the name of file containing fortin namelist.
  integer, parameter :: maxlen = 256
  character :: probin*(maxlen)

  if (namlen .gt. maxlen) then
     call castro_error('probin file name too long')
  end if

  do i = 1, namlen
     probin(i:i) = char(name(i))
  end do

  ! Set namelist defaults
  rho = 1.e0_rt
  temp = 3.e2_rt
  vel = 1.e3_rt

  ! read namelists
  untin = 9
  open(untin,file=probin(1:namlen),form='formatted',status='old')
  read(untin,fortin)
  close(unit=untin)

  eos_state % rho = rho
  eos_state % T = temp
  eos_state % xn = 0.e0_rt
  eos_state % xn(1) = 1.e0_rt
  
  call eos(eos_input_rt, eos_state)
  
  rhoe = rho * eos_state % e

end subroutine amrex_probinit

subroutine ca_initdata(level, time, lo, hi, nscal, &
                       state, s_lo, s_hi, &
                       dx, xlo, xhi)

  use network, only: nspec
  use probdata_module
  use meth_params_module, only: NVAR, URHO, UMX, UMY, UMZ, UEDEN, UEINT, UTEMP, UFS
  use amrex_fort_module, only: rt => amrex_real

  implicit none

  integer,  intent(in) :: level, nscal
  integer,  intent(in) :: lo(3), hi(3)
  integer,  intent(in) :: s_lo(3), s_hi(3)
  real(rt), intent(in) :: xlo(3), xhi(3), time, dx(3)
  
  real(rt), intent(inout) :: state(s_lo(1):s_hi(1), s_lo(2):s_hi(2), s_lo(3):s_hi(3), NVAR)

  integer :: i, j, k
  real(rt) :: x, y, r, cos_phi, sin_phi

  do k = lo(3), hi(3)
    
    do j = lo(2), hi(2)
      y = xlo(2) + dx(2)*(float(j-lo(2)) + 0.5e0_rt)
      
      do i = lo(1), hi(1)
        x = xlo(1) + dx(1)*(float(i-lo(1)) + 0.5e0_rt)
        
        r = sqrt(x**2 + y**2)
        cos_phi = x/r
        sin_phi = y/r
        
        ! Want momentum vector to point in the \hat{\phi} direction
        ! rho * vel * (-sin_phi * \hat{x} + cos_phi * \hat{y})
        state(i, j, k, UMX) = -sin_phi * rho * vel
        state(i, j, k, UMY) = cos_phi * rho * vel
        state(i, j, k, UMZ) = 0.e0_rt
        
        ! Other state components
        state(i, j, k, URHO) = rho
        state(i, j, k, UEDEN) = rhoe + 0.5*rho*vel*vel
        state(i, j, k, UEINT) = rhoe
        state(i, j, k, UTEMP) = temp
        
        state(i, j, k, UFS:UFS-1+nspec) = 0.0e0_rt
        state(i, j, k, UFS) = rho
        
      end do
    end do
  end do

end subroutine ca_initdata
