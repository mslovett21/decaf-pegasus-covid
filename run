#!/bin/bash

# Parameters
NWORKERS=$1
NTASKS=$(($1+1))
TIMES=$2
NTRIALS=256
NCORES=64
CP_FREQ=200000

# Prepare covid.conf
rm covid-${NWORKERS}.conf
for (( i=1; i<=$NWORKERS; i++ ))
do
    echo " $(($i-1)) python ./worker.py --epochs 1 --ex_rate 1 --cuda 0 " >> covid-${NWORKERS}.conf
done
# Master task
echo " $(($i-1)) python ./master.py $NTRIALS 1 $CP_FREQ " >> covid-${NWORKERS}.conf

# Prepare the workflow graph (optuna.json)
cp optuna-covid19.py optuna-${NWORKERS}.py
# TODO sed trick for optuna-${NWORKERS}.py
sed -i 's/0, nprocs=2/0, nprocs='"$NWORKERS"'/g' optuna-${NWORKERS}.py
sed -i 's/n", start_proc=2/n", start_proc='"$NWORKERS"'/g' optuna-${NWORKERS}.py
python3 optuna-${NWORKERS}.py

for (( i=1; i<=$TIMES; i++ ))
do
    echo "--- Start trial $i with Decaf"
    # Cleanup
    rm hpo_study_checkpoint_* logs/*

    # Execution
    echo "--- Starting $NWORKERS worker, each worker runs on $NCORES cores and explores ${NTRIALS} trials in total"
    start=$SECONDS
    srun -n ${NTASKS} -c ${NCORES} --multi-prog covid-${NWORKERS}.conf &> log.decaf.${NWORKERS}_${i}
    end=$SECONDS
    echo "Duration: $((end-start)) seconds." >> log.decaf.${NWORKERS}_${i}
    echo "--- End trial $i with Decaf"
done
