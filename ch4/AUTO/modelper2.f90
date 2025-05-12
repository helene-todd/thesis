!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 
!   modelper :   model by Todd, Gutkin & Cayco Gajic, 2024
!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 

SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!--------- ---- 

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,*), DFDP(NDIM,*)

  DOUBLE PRECISION R1, R2, V1, V2, S1, S2
  DOUBLE PRECISION G1, J2, A, TAUM, TAUD, ETA, DELTA, J1, G2, JC, T, PI
  
       ! DEFINE THE STATE VARIABLES
       R1  = U(1)
       R2  = U(2)
       V1  = U(3)
       V2  = U(4)
       S1  = U(5)
       S2  = U(6)
       
       ! DEFINE THE SYSTEM PARAMETERS
       G1    = PAR(1)
       J2    = PAR(2)
       A     = PAR(3)
       TAUM  = PAR(4)
       TAUD  = PAR(5)
       ETA   = PAR(6)
       DELTA = PAR(7)
       J1    = PAR(8)
       G2    = PAR(9)
       JC    = PAR(10)
       T     = PAR(11)


       ! DEFINE PI
       PI = 4 * ATAN(1.0d0)

       
       ! DEFINE THE RIGHT-HAND SIDE
       F(1) = (DELTA/(TAUM*PI) + 2*R1*V1 - 2*TAUM*LOG(A)*R1**2 - G1*R1)/TAUM
       F(2) = (DELTA/(TAUM*PI) + 2*R2*V2 - 2*TAUM*LOG(A)*R2**2 - G2*R2)/TAUM
       F(3) = (V1**2 + ETA + J1*TAUD*S1 + JC*TAUD*S2 + (DELTA*LOG(A))/PI - (LOG(A)**2+PI**2)*(TAUM*R1)**2)/TAUM
       F(4) = (V2**2 + ETA + J2*TAUD*S2 + JC*TAUD*S1 + (DELTA*LOG(A))/PI - (LOG(A)**2+PI**2)*(TAUM*R2)**2)/TAUM
       F(5) = (-S1+R1)/TAUD
       F(6) = (-S2+R2)/TAUD
       
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
      PAR(1)  =   0.800d0           ! PARAMETER G1
      PAR(2)  =  -3.200d0           ! PARAMETER J2
      PAR(3)  =   1.000d0           ! PARAMETER A
      PAR(4)  =   1.000d0           ! PARAMETER TAUM
      PAR(5)  =   1.000d0           ! PARAMETER TAUD
      PAR(6)  =   1.000d0           ! PARAMETER ETA
      PAR(7)  =   0.300d0           ! PARAMETER DELTA
      PAR(8)  =  -2.500d0           ! PARAMETER J1
      PAR(9)  =   2.000d0           ! PARAMETER G2
      PAR(10) =  -8.000d0           ! PARAMETER JC
      PAR(11) =   6.800d0           ! PERIOD

      ! VARIABLES
     ! U(1) =  0.222286d0  ! VARIABLE R1
     ! U(2) =  0.218301d0 ! VARIABLE R2
     ! U(3) =  0.18547d0  ! VARIABLE V1
     ! U(4) =  1.52604d0  ! VARIABLE V2
     ! U(5) =  0.158287d0  ! VARIABLE S1
     ! U(6) =  0.113232d0 ! VARIABLE S2

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
