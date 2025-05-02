SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!--------- ---- 

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,*), DFDP(NDIM,*)

  DOUBLE PRECISION R, V, S
  DOUBLE PRECISION G, J, A, TAUM, TAUD, ETA, DELTA, T, PI
  
       ! DEFINE THE STATE VARIABLES
       R   = U(1)
       V   = U(2)
       
       ! DEFINE THE SYSTEM PARAMETERS
       G     = PAR(1)
       J     = PAR(2)
       A     = PAR(3)
       TAUM  = PAR(4)
       ETA   = PAR(5)
       DELTA = PAR(6)
       T     = PAR(11)


       ! DEFINE PI
       PI = 4 * ATAN(1.0d0)

       
       ! DEFINE THE RIGHT-HAND SIDE
       F(1) = (DELTA/(TAUM*PI) + 2*R*V - 2*TAUM*LOG(A)*R**2 - G*R)/TAUM
       F(2) = (V**2 + ETA + J*TAUM*R + (DELTA*LOG(A))/PI - (LOG(A)**2+PI**2)*(TAUM*R)**2)/TAUM
       
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
      PAR(1)  =   0.400d0           ! PARAMETER G
      PAR(2)  =   10.000d0          ! PARAMETER J
      PAR(3)  =   1.000d0           ! PARAMETER A
      PAR(4)  =   1.000d0           ! PARAMETER TAUM
      PAR(5)  =   -1.000d0           ! PARAMETER ETA
      PAR(6)  =   0.300d0           ! PARAMETER DELTA
      PAR(11)  =  3.270d0

      ! VARIABLES
      U(1) =  1.16027    ! VARIABLE R 
      U(2) =  2.35885    ! VARIABLE V
        

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
