#!/usr/bin/env python3

import pybredala as bd
import pydecaf as d
from mpi4py import MPI

import argparse
import time
import os,sys
import joblib
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

import utils.util as util
from trainer.train import initialize, train, validation
from IPython import embed

import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)

### -------------------------VARIABLES--------------------------------

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL      = 'COVIDNet_small'
EPOCHS     = 10
BATCH_SIZE = 12
WORKER_ID  = 0
STUDY      = None
EXCHANGE_RATE = 2
OWN_NEW_TRIALS    = 0
ARGS = ""

### ------------------------- PARSER --------------------------------
def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=500, help='steps to print metrics and loss')
    parser.add_argument('--dataset_name', type=str, default='COVIDx', help='dataset name COVIDx')
    parser.add_argument('--cuda', type=int, default=0, help='use gpu support')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet_small',choices=('COVIDNet_small', 'resnet18', 'COVIDNet_large'))
    parser.add_argument('--root_path', type=str, default='./data',help='path to dataset ')
    parser.add_argument('--save', type=str, default='./saved/COVIDNet' + util.datestr(),help='path to checkpoint save directory ')
    parser.add_argument('--epochs', type=int,default=1, help = "number of training epochs")
    parser.add_argument('--trials', type=int, default=2, help = "number of HPO trials")
    parser.add_argument('--ex_rate',type=int,default=2, help = "info exchange rate in HPO")

    args = parser.parse_args()
    return args

### -------------------------HPO--------------------------------

def objective(trial):

    model, training_generator, val_generator, test_generator = initialize(ARGS)

    optim_name   = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    lr           = trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True)
    trial.set_user_attr("worker_id", WORKER_ID)

    optimizer = util.select_optimizer(optim_name, model, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5, verbose=True)

    best_pred_loss = 1000.0

    for epoch in range(1, EPOCHS + 1):
        train(ARGS, model, training_generator, optimizer, epoch)
        val_metrics, confusion_matrix = validation(ARGS, model, val_generator, epoch)
        scheduler.step(val_metrics._data.average.loss)

    return val_metrics._data.average.recall_mean

def hpo_monitor(study):
    joblib.dump(study,"hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID))

def create_study(hpo_checkpoint_file, decaf):
    global STUDY

    STUDY = optuna.create_study(direction = 'maximize', study_name = MODEL, pruner=optuna.pruners.NopPruner())
    STUDY.set_user_attr("worker_id", WORKER_ID)

    container_in = bd.pSimple() #container_in: decaf data container for data exchange between master and workers.
    rate = EXCHANGE_RATE #this is here for the initial run, then exchange_rate will be governed by the master.
    while(decaf.get(container_in, "in")):
        #orc: cycle logic.
        if container_in.get().isToken() :
            print("received token")
            container_in.get().setToken(False)
        else:
            #receive the trial_info from master
            recv = container_in.get().getFieldDataSI("rate")
            rate = recv.getData()
            new_trial = container_in.get().getFieldDataVF("out_trial")
            nt_values = new_trial.getVector()
            print("worker at rank " + str(WORKER_ID) + " received "+ str(rate) + " length of " + str(len(nt_values)))
            #orc: manipulating the new_trials to remove our trial values
            import numpy as np
            X = np.array(nt_values)
            arr_size = int(len(nt_values)/5)
            trial_5D = X.reshape((arr_size, 5))

            for i in range(arr_size):
                if trial_5D[i][4] != WORKER_ID :
                    if trial_5D[i][1] == 0 :
                        opt_str = "Adam"
                    elif trial_5D[i][1] == 1 :
                        opt_str = "RMSprop"
                    else:
                        opt_str = "SGD"
                    new_trial = optuna.trial.create_trial(
                        params = {"weight_decay": trial_5D[i][0] ,"optimizer": opt_str, "learning_rate":trial_5D[i][2] },
                        distributions = {"weight_decay": LogUniformDistribution(1e-5, 1e-1),
                        "optimizer" : CategoricalDistribution(choices=('Adam', 'RMSprop', 'SGD')),
                        "learning_rate": LogUniformDistribution(1e-7, 1e-5)},
                        value= trial_5D[i][3],)
                    STUDY.add_trial(new_trial)
                    print("New trial was added at worker at rank " + str(WORKER_ID))

        print("performing more " + str(rate) + " trials...")
        if rate > 0 :
            STUDY.optimize(objective, n_trials=rate, timeout=600)

        print("Study statistics: ")
        print("Number of finished trials: ", len(STUDY.trials))
        print("Best trial:")
        trial = STUDY.best_trial
        print(trial)
        print("  Value: ", trial.value)

        if (rate > 0):
            #orc: send the trial_info to the master
            completed_trials = STUDY.trials[-rate:]
            trial_values = []
            for trial in completed_trials:
                dpv = trial.params['weight_decay']
                opt = trial.params['optimizer']
                if opt == "Adam" :
                    opt_ctg = 0
                elif opt == "RMSprop" :
                    opt_ctg = 1
                else:
                    opt_ctg = 2
                lrv = trial.params['learning_rate']
                acv = trial.value
                trial_values.append(dpv)
                trial_values.append(opt_ctg)
                trial_values.append(lrv)
                trial_values.append(acv)
                trial_values.append(WORKER_ID)

            data_trial = bd.VectorFieldf(trial_values,rate*5)
            container_out = bd.pSimple()
            container_out.get().appendData("trial", data_trial, bd.DECAF_NOFLAG, bd.DECAF_PRIVATE, bd.DECAF_SPLIT_DEFAULT, bd.DECAF_MERGE_APPEND_VALUES)
            decaf.put(container_out,"out")

    print("worker " + str(WORKER_ID) + " terminating")
    decaf.terminate()

### -------------------------MAIN--------------------------------
def main():

    global MODEL
    global EPOCHS
    global WORKER_ID
    global EXCHANGE_RATE
    global ARGS

    w = d.Workflow()
    w.makeWflow(w,"optuna.json")

    a = MPI._addressof(MPI.COMM_WORLD)
    decaf = d.Decaf(a,w)
    r = MPI.COMM_WORLD.Get_rank()

    try:
        ARGS = get_arguments()
        SEED = ARGS.seed

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        if (ARGS.cuda):
            torch.cuda.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        EPOCHS        = ARGS.epochs
        WORKER_ID     = r
        EXCHANGE_RATE = ARGS.ex_rate
        MODEL         = ARGS.model

        hpo_checkpoint_file = "hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID)

        create_study(hpo_checkpoint_file, decaf)

    except Exception as e:
        print(e)
    finally:
        hpo_monitor(STUDY)
    return 0


if __name__ == '__main__':
    main()




