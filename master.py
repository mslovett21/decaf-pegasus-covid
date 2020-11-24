#!/usr/bin/env python3

import pybredala as bd
import pydecaf as d
from mpi4py import MPI

import argparse
import os,sys

import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)

import joblib
### ------------------------- PARSER --------------------------------
arch_names = ["basicnet", "densenet121", "vgg16"]

parser = argparse.ArgumentParser(description='Manager for sharing HPO results between workers')
parser.add_argument('trials',  metavar='num_trials', type=int, nargs=1, help = "number of HPO trials")
parser.add_argument('ex_rate',  metavar='exchange_rate', type=int, nargs=1, help = "info exchange rate in HPO",  default=2)
parser.add_argument('cp_freq',  metavar='checkpoint_freq', type=int, nargs=1, help = "checkpoint frequency",  default=16)
parser.add_argument('--model', type=str, default='COVIDNet_small',choices=('COVIDNet_small', 'resnet18', 'COVIDNet_large'))

### -------------------------MAIN--------------------------------
def main():

    global TOTAL_TRIALS
    global CHECKPOINT_FREQ
    global MODEL

    w = d.Workflow()
    w.makeWflow(w,"optuna.json")

    a = MPI._addressof(MPI.COMM_WORLD)
    decaf = d.Decaf(a,w)
    r = MPI.COMM_WORLD.Get_rank()


    args            = parser.parse_args()
    TOTAL_TRIALS    = args.trials[0]
    EXCHANGE_RATE   = args.ex_rate[0]
    CHECKPOINT_FREQ = args.cp_freq[0]
    MODEL           = args.model

    print("Total number of trials is {}".format(TOTAL_TRIALS))
    print("Exchange rate is {}".format(EXCHANGE_RATE))

    #orc@03-09: creating a study object at the master to checkpoint at the master:
    STUDY = optuna.create_study(direction = 'maximize', study_name = MODEL, pruner=optuna.pruners.NopPruner())
    STUDY.set_user_attr("worker_id", r)

    container = bd.pSimple()
    iter = 0
    while(TOTAL_TRIALS > 0):
        #orc: fetch the trial_info from workers
        decaf.get(container, "in")
        in_trial = container.get().getFieldDataVF("trial")
        trial_values = in_trial.getVector()
        print("master at rank " + str(r) + " received data of length" + str(len(trial_values)))

        #orc: adding the received trials to the study object:
        import numpy as np
        X = np.array(trial_values)
        arr_size = int(len(trial_values)/5)
        trial_5D = X.reshape((arr_size, 5))
        for i in range(arr_size):
            if trial_5D[i][1] == 0 :
                opt_str = "Adam"
            elif trial_5D[i][1] == 1 :
                opt_str = "RMSprop"
            else:
                opt_str = "SGD"
            new_trial = optuna.trial.create_trial(
                params = {"weight_decay": trial_5D[i][0] ,"optimizer": opt_str, "lr":trial_5D[i][2] },
                distributions = {"weight_decay": LogUniformDistribution(1e-5, 1e-1),
                "optimizer" : CategoricalDistribution(choices=('Adam', 'RMSprop', 'SGD')),
                "lr": LogUniformDistribution(1e-7, 1e-5)},
                value= trial_5D[i][3],)
            STUDY.add_trial(new_trial)
            #print("New trial was added at MASTER at rank " + str(r))

        #orc: printing study stats on master:
        print("Study statistics at MASTER: ")
        print("Number of finished trials at MASTER: ", len(STUDY.trials))
        print("Best trial at MASTER:")
        trial = STUDY.best_trial
        print(trial)
        print("  Value: ", trial.value)
        #orc: checkpoint in a given interval by the user
        iter+=int(len(trial_values)/5)
        if iter >= CHECKPOINT_FREQ:
            joblib.dump(STUDY,"hpo_study_checkpoint_MASTER_{}_{}.pkl".format(MODEL, len(STUDY.trials)))
            iter = 0

        #TODO check whether this will create a problem when there is pruning.
        TOTAL_TRIALS-= int(len(trial_values)/5)
        print("remained this many trials at MASTER : " + str(TOTAL_TRIALS))
        rate = min(TOTAL_TRIALS, EXCHANGE_RATE)
        #orc: send back the trial_info to the workers, together with the communication frequency we want.
        out_trial = bd.VectorFieldf(trial_values, 5)
        data = bd.SimpleFieldi(rate)
        container_out = bd.pSimple()
        container_out.get().appendData("out_trial", out_trial, bd.DECAF_NOFLAG, bd.DECAF_PRIVATE, bd.DECAF_SPLIT_KEEP_VALUE, bd.DECAF_MERGE_DEFAULT)
        container_out.get().appendData("rate", data, bd.DECAF_NOFLAG, bd.DECAF_PRIVATE, bd.DECAF_SPLIT_KEEP_VALUE, bd.DECAF_MERGE_DEFAULT)
        decaf.put(container_out,"out")

    print("master at rank " + str(r) + " terminating")
    decaf.terminate()

    return 0


if __name__ == '__main__':
    main()



