module bc_ext_fill_module

  use amrex_fort_module, only : rt => amrex_real
  implicit none

  public

contains

  ! ::: -----------------------------------------------------------

  subroutine ext_fill(lo, hi, adv, adv_lo, adv_hi, &
                      domlo, domhi, delta, xlo, time, bc) bind(C, name="ext_fill")

    use meth_params_module, only : NVAR, URHO, UMX, UMY, UMZ, UEDEN, UEINT, UFS, UTEMP
    use amrex_fort_module, only : rt => amrex_real
    
    implicit none

    include 'AMReX_bc_types.fi'

    integer, intent(in) :: lo(3), hi(3)
    integer, intent(in) :: adv_lo(3), adv_hi(3)
    integer, intent(in) :: domlo(3), domhi(3)
    integer, intent(in) :: bc(AMREX_SPACEDIM, 2, NVAR)
    
    real(rt), intent(in) :: delta(3), xlo(3)
    real(rt), intent(inout) :: adv(adv_lo(1):adv_hi(1), adv_lo(2):adv_hi(2), &
                                   adv_lo(3):adv_hi(3), NVAR)
    real(rt), intent(in), value :: time

    integer i, j, k, n
    
    ! Need to reflect over the line y = x and fill with the values
    ! along that boundary.
    
    !$gpu

    ! XLO
    if(bc(1, 1, 1) .eq. EXT_DIR .and. adv_lo(1) .lt. domlo(1)) then
       do k = adv_lo(3), adv_hi(3)
          do j = adv_lo(2), adv_hi(2)
             do i = adv_lo(1), domlo(1)-1
                adv(i, j, k, URHO) = adv(j, i+1, k, URHO)
                write(*,*) adv(i, j, k, URHO)
                adv(i, j, k, UMX) = -adv(j, i+1, k, UMY)
                adv(i, j, k, UMY) = adv(j, i+1, k, UMX)
                adv(i, j, k, UMZ) = adv(j, i+1, k, UMZ)
                adv(i, j, k, UFS) = adv(j, i+1, k, URHO)
                adv(i, j, k, UEINT) = adv(j, i+1, k, UEINT)
                adv(i, j, k, UEDEN) = adv(j, i+1, k, UEDEN)
                adv(i, j, k, UTEMP) = adv(j, i+1, k, UTEMP)
             end do
          end do
       end do
    end if

    ! XHI
    if(bc(1, 2, 1) .eq. EXT_DIR .and. adv_hi(1) .gt. domhi(1)) then
       do k = adv_lo(3), adv_hi(3)
          do j = adv_lo(2), adv_hi(2)
             do i = domhi(1)+1, adv_hi(1)
                 adv(i, j, k, URHO) = adv(j, i-1, k, URHO)
                 adv(i, j, k, UMX) = -adv(j, i-1, k, UMY)
                 adv(i, j, k, UMY) = adv(j, i-1, k, UMX)
                 adv(i, j, k, UMZ) = adv(j, i-1, k, UMZ)
                 adv(i, j, k, UFS) = adv(j, i-1, k, URHO)
                 adv(i, j, k, UEINT) = adv(j, i-1, k, UEINT)
                 adv(i, j, k, UEDEN) = adv(j, i-1, k, UEDEN)
                 adv(i, j, k, UTEMP) = adv(j, i-1, k, UTEMP)
             end do
          end do
       end do
    end if

    ! YLO
    if(bc(2, 1, 1) .eq. EXT_DIR .and. adv_lo(2) .lt. domlo(2)) then
       do k = adv_lo(3), adv_hi(3)
          do j = adv_lo(2), domlo(2)-1
             do i = adv_lo(1), adv_hi(1)
                 adv(i, j, k, URHO) = adv(j+1, i, k, URHO)
                 adv(i, j, k, UMX) = -adv(j+1, i, k, UMY)
                 adv(i, j, k, UMY) = adv(j+1, i, k, UMX)
                 adv(i, j, k, UMZ) = adv(j+1, i, k, UMZ)
                 adv(i, j, k, UFS) = adv(j+1, i, k, URHO)
                 adv(i, j, k, UEINT) = adv(j+1, i, k, UEINT)
                 adv(i, j, k, UEDEN) = adv(j+1, i, k, UEDEN)
                 adv(i, j, k, UTEMP) = adv(j+1, i, k, UTEMP)
             end do
          end do
       end do
    end if

    ! YHI
    if(bc(2, 2, 1) .eq. EXT_DIR .and. adv_hi(2) .gt. domhi(2)) then
       do k = adv_lo(3), adv_hi(3)
          do j = domhi(2)+1, adv_hi(2)
             do i = adv_lo(1), adv_hi(1)
                 adv(i, j, k, URHO) = adv(j-1, i, k, URHO)
                 adv(i, j, k, UMX) = -adv(j-1, i, k, UMY)
                 adv(i, j, k, UMY) = adv(j-1, i, k, UMX)
                 adv(i, j, k, UMZ) = adv(j-1, i, k, UMZ)
                 adv(i, j, k, UFS) = adv(j-1, i, k, URHO)
                 adv(i, j, k, UEINT) = adv(j-1, i, k, UEINT)
                 adv(i, j, k, UEDEN) = adv(j-1, i, k, UEDEN)
                 adv(i, j, k, UTEMP) = adv(j-1, i, k, UTEMP)
             end do
          end do
       end do
    end if

  end subroutine ext_fill

  ! :::
  ! ::: -----------------------------------------------------------
  ! :::

  subroutine ext_denfill(lo, hi, adv, adv_lo, adv_hi, &
                         domlo, domhi, delta, xlo, time, bc) bind(C, name="ext_denfill")

    use probdata_module

    implicit none

    include 'AMReX_bc_types.fi'

    integer,  intent(in) :: lo(3), hi(3)
    integer,  intent(in) :: adv_lo(3), adv_hi(3)
    integer,  intent(in) :: domlo(3), domhi(3)
    integer,  intent(in) :: bc(AMREX_SPACEDIM, 2)
    real(rt), intent(in) :: delta(3), xlo(3)
    real(rt), intent(inout) :: adv(adv_lo(1):adv_hi(1), adv_lo(2):adv_hi(2), adv_lo(3):adv_hi(3))
    real(rt), intent(in), value :: time

    integer :: i, j, k

    !$gpu

    ! XLO
    if(bc(1, 1) .eq. EXT_DIR .and. adv_lo(1) .lt. domlo(1)) then
       do k = adv_lo(3), adv_hi(3)
          do j = adv_lo(2), adv_hi(2)
             do i = adv_lo(1), domlo(1)-1
                adv(i, j, k) = adv(j, i+1, k)
             end do
          end do
       end do
    end if

    ! XHI
    if(bc(1, 2) .eq. EXT_DIR .and. adv_hi(1) .gt. domhi(1)) then
       do k = adv_lo(3), adv_hi(3)
          do j = adv_lo(2), adv_hi(2)
             do i = domhi(1)+1, adv_hi(1)
                adv(i, j, k) = adv(j, i-1, k)
             end do
          end do
       end do
    end if

    ! YLO
    if(bc(2, 1) .eq. EXT_DIR .and. adv_lo(2) .lt. domlo(2)) then
       do k = adv_lo(3), adv_hi(3)
          do j = adv_lo(2), domlo(2)-1
             do i = adv_lo(1), adv_hi(1)
                adv(i, j, k) = adv(j+1, i, k)
             end do
          end do
       end do
    end if

    ! YHI
    if(bc(2, 2) .eq. EXT_DIR .and. adv_hi(2) .gt. domhi(2)) then
       do k = adv_lo(3), adv_hi(3)
          do j = domhi(2)+1, adv_hi(2)
             do i = adv_lo(1), adv_hi(1)
                adv(i, j, k) = adv(j-1, i, k)
             end do
          end do
       end do
    end if

  end subroutine ext_denfill

end module bc_ext_fill_module
