#!/bin/bash
#SBATCH -A m2187
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -n 8
#SBATCH -c 4
#SBATCH --ntasks-per-node=8
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3,4,5,6,7
#SBATCH -t 04:00:00

if [ "$#" -ne 4 ]; then
    echo "./main.sh NTRIALS NWORKERS NTHREADS_PER_WORKER NJOBS"
    exit 2
fi

# Parameters
NTRIALS=$1
NWORKERS=$2
NTHREADS_PER_WORKER=$3
NJOBS=$4

#Change to desired number of times each version should run
NTIMES=2

echo "Number of trials: ${NTRIALS}"
echo "Number of workers: ${NWORKERS}"
echo "Number of threads per worker: ${NTHREADS_PER_WORKER}"
echo "Number of running times: ${NTIMES}"
echo "Number of Python threads: ${NJOBS}"

# Load environment script
. $HOME/application/envs/decaf.sh

# DB version
echo "Run database version"
cd db
./run ${NTRIALS} ${NWORKERS} ${NTHREADS_PER_WORKER} ${NJOBS} ${NTIMES}
cd .. 

# Decaf version
#echo "Run Decaf version with ${NWORKERS} workers in ${NTIMES} times"
#cd decaf
#./run ${NTRIALS} ${NWORKERS} ${NTHREADS_PER_WORKER} ${NTIMES}
#cd ..
