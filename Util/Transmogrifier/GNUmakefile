CASTRO_HOME := ../..

PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = TRUE

USE_GRAV   = FALSE
USE_REACT  = FALSE

EOS_DIR     := helmholtz
NETWORK_DIR := general_null
NETWORK_INPUTS = ignition_wdconvect.net

USE_TRANSMOGRIFIER = TRUE
USE_PERMISSIVE_SPACEDIM = TRUE

Bpack   := ./Make.package
Blocs   := .

include $(CASTRO_HOME)/Exec/Make.Castro
