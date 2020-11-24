#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --constraint=haswell

# Parameters
NWORKERS=$1
NTIMES=2  #Change to desired number of times each version should run

# Load environment script
. $HOME/application/envs/pegasus.sh

# DB version
echo "Run database version with ${NWORKERS} workers in ${NTIMES} times"
cd db
./run ${NWORKERS} ${NTIMES}
cd .. 

# Decaf version
echo "Run Decaf version with ${NWORKERS} workers in ${NTIMES} times"
cd decaf
./run ${NWORKERS} ${NTIMES}
cd ..
