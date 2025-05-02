SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!--------- ---- 

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,*), DFDP(NDIM,*)

  DOUBLE PRECISION R, V
  DOUBLE PRECISION G, J, ETA, T, PI
  
       ! DEFINE THE STATE VARIABLES
       R   = U(1)
       V   = U(2)
       
       ! DEFINE THE SYSTEM PARAMETERS
       G     = PAR(1)
       J     = PAR(2)
       ETA   = PAR(3)
       T     = PAR (11)


       ! DEFINE PI
       PI = 4 * ATAN(1.0d0)

       
       ! DEFINE THE RIGHT-HAND SIDE
       F(1) = 1 + 2*R*V - G*R
       F(2) = V**2 + ETA + J*R - R**2
       
END SUBROUTINE FUNC
!---------------------------------------------------------------------- 

SUBROUTINE STPNT(NDIM,U,PAR,T)

!     ---------- ----- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION TPI

      ! PARAMETERS
      PAR(1)  =   1.000d0           ! PARAMETER G
      PAR(2)  =   3.000d0           ! PARAMETER J
      PAR(3)  =   -2.000d0          ! PARAMETER ETA
      
      PAR(11) =   3.060d0 

      ! VARIABLES
      U(1) =  2.51309     ! VARIABLE R 
      U(2) =  0.244632    ! VARIABLE V
      
      
END SUBROUTINE STPNT
!---------------------------------------------------------------------- 

SUBROUTINE PVLS
END SUBROUTINE PVLS

SUBROUTINE BCND
END SUBROUTINE BCND

SUBROUTINE ICND 
END SUBROUTINE ICND

SUBROUTINE FOPT 
END SUBROUTINE FOPT
!---------------------------------------------------------------------- 
