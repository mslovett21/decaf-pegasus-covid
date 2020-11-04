#!/bin/bash
# Loaded Modules
module purge
module load PrgEnv-gnu
module load craype-haswell
module load cray-mpich
module load python
module load cmake
module load boost/1.69.0
module load git

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic
