#!/bin/bash
# Loaded Modules
module purge
unset LD_LIBRARY_PATH
module load PrgEnv-gnu
module load craype-haswell
module load cray-mpich
module load python
module load cmake
module load boost/1.69.0
module load cuda
module load git
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
#source activate pegasus_env
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib64:/usr/lib64:$HOME/.conda/envs/pegasus_env/lib"

# Decaf
export DECAF_PREFIX="${HOME}/software/install/decaf"
export LD_LIBRARY_PATH="${DECAF_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic
