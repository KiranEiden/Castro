PRECISION = DOUBLE
PROFILE = FALSE

DEBUG = FALSE

DIM = 2
COMP = gnu
USE_CUDA = FALSE

USE_MPI = TRUE
USE_OMP = TRUE
USE_RAD = FALSE
USE_GRAV = TRUE
USE_POINTMASS = TRUE
USE_NEUTRINO = FALSE
USE_MASS_SOURCE = TRUE
USE_SIN_DEP = FALSE # sin^2(theta) variation
USE_COS_DEP = FALSE # cos^2(theta) variation
USE_EQ_DEP = FALSE # Deposit within some equatorial angle
USE_SPECIES_SOURCES = TRUE

CASTRO_HOME = ../../..
EOS_DIR     := gamma_law

NETWORK_DIR := general_null
NETWORK_INPUTS := mag_sup_small.net
#Opacity_dir = csm
Bpack   := ./Make.package
Blocs   := .

include $(CASTRO_HOME)/Exec/Make.Castro
