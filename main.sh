#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=3
#SBATCH --constraint=haswell

if [ "$#" -ne 1 ]; then
    echo "./main.sh number_of_workers"
    exit 2
fi

# Parameters
NWORKERS=$1

#Change to desired number of times each version should run
NTIMES=2

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
