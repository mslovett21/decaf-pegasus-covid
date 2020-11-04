#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --constraint=haswell

source ./env/init.sh
NWORKERS=$1
NCORES=$2
NTRIALS=$3
echo "-- Starting $NWORKERS worker, each worker runs on $NCORES cores and explores $NTRIALS"
srun -n $NWORKERS -c $NCORES python main_optuna_db.py --trials $NTRIALS 
