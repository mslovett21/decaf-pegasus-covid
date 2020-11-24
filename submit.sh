#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=3
#SBATCH --constraint=haswell

# Parameters
NWORKERS=$1
NTIMES=1  #Change to desired number of times each version should run

# Load environment script
module restore decaf-tu

# Decaf version
echo "Run Decaf version with ${NWORKERS} workers in ${NTIMES} times"
#cd decaf-pegasus
./run.sh ${NWORKERS} ${NTIMES}
