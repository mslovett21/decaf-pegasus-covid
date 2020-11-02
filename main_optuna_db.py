import argparse
import time
import joblib
import numpy as np
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

import utils.util as util
from trainer.train import initialize, train, validation
from IPython import embed

import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.ERROR)

import time
from mpi4py import MPI

### ------------------------- LOGGER--------------------------------
logger = logging.getLogger('optuna_db_log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ARGS = ""
MODEL = 'COVIDNet_small'
EPOCHS = 10
LOG_DIR = 'logs/'
WORKER_ID = 0 
BATCH_SIZE = 12
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=12, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=200, help='steps to print metrics and loss')
    parser.add_argument('--cuda', type=int, default=0, help='use gpu support')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--dataset_name', type=str, default='COVIDx', help='dataset name COVIDx')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet_small',choices=('COVIDNet_small', 'resnet18', 'COVIDNet_large'))
    parser.add_argument('--root_path', type=str, default='./data',help='path to dataset ')
    parser.add_argument('--save', type=str, default='./saved/COVIDNet' + util.datestr(),help='path to checkpoint save directory ')
    parser.add_argument('--epochs', type=int,default=1, help = "number of training epochs")
    parser.add_argument('--trials', type=int, default=4, help = "number of HPO trials")
    parser.add_argument('--worker_id', type=int, default=0, help = "worker id")
    parser.add_argument('--ex_rate',type=int,default=2, help = "info exchange rate in HPO")
    
    args = parser.parse_args()
    
    return args

def objective(trial):
    start = time.time()
    model, training_generator, val_generator, test_generator = initialize(ARGS)

    optim_name   = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    lr           = trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True)
    #trial.set_user_attr("worker_id", WORKER_ID)
    
    optimizer = util.select_optimizer(optim_name, model, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5, verbose=True)

    best_pred_loss = 1000.0
    epochs = ARGS.epochs
        
    for epoch in range(1, epochs + 1):
        
        train(ARGS, model, training_generator, optimizer, epoch)        
        val_metrics, confusion_matrix = validation(ARGS, model, val_generator, epoch)
        scheduler.step(val_metrics._data.average.loss)

    print(f'Worker {WORKER_ID} finished trial {trial.number} in {time.time() - start} seconds')
    return val_metrics._data.average.recall_mean


def hpo_monitor(study,trial):
    joblib.dump(study,"hpo_study_checkpoint_optuna_{}_{}.pkl".format(MODEL, WORKER_ID))


def final_hpo_monitor(study):
    joblib.dump(study,"hpo_study_checkpoint_optuna_{}_{}.pkl".format(MODEL, WORKER_ID))



def create_study(hpo_checkpoint_file, total_trials):
   
    try:
        study = joblib.load("hpo_study_checkpoint_optuna_{}_{}.pkl".format(MODEL, WORKER_ID))
        todo_trials = total_trials - len(study.trials_dataframe())
        if todo_trials > 0 :
            logger.info("There are {} trial(s) to do out of {}".format(todo_trials, total_trials))
            print("There are {} trial(s) to do out of {}".format(todo_trials, total_trials))
            study.optimize(objective, n_trials=todo_trials, timeout=600, callbacks=[hpo_monitor])
    except:
        #study = optuna.create_study(direction = 'maximize', study_name = MODEL, storage='mysql://decaf_hpo_db_admin:3Iiidd_2s25j3w33jjdd@nerscdb04.nersc.gov/decaf_hpo_db')
        study = optuna.load_study(study_name = MODEL, storage='mysql://decaf_hpo_db_admin:3Iiidd_2s25j3w33jjdd@nerscdb04.nersc.gov/decaf_hpo_db')
        #study.set_user_attr("worker_id", WORKER_ID)
        #study.optimize(objective, n_trials=total_trials, timeout=600, callbacks=[hpo_monitor])
        study.optimize(objective, n_trials=total_trials)
    return study



def main():
    
    global MODEL
    global WORKER_ID
    global ARGS
    
    ARGS = get_arguments()   
    seed = ARGS.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)

       
    if (ARGS.cuda):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   
    
    total_trials  = ARGS.trials
    WORKER_ID     = MPI.COMM_WORLD.Get_rank()
    MODEL         = ARGS.model

    fh = logging.FileHandler(LOG_DIR + 'main_worker_{}.log'.format(WORKER_ID))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    try:
        hpo_checkpoint_file = "hpo_study_checkpoint_optuna_{}_{}.pkl".format(MODEL, WORKER_ID)
        study = create_study(hpo_checkpoint_file, total_trials)   
    except Exception as e:
        logger.info(e)    
    finally:
        #MPI.COMM_WORLD.Barrier()
        final_hpo_monitor(study)
        print(study.trials_dataframe())

    return 0


if __name__ == '__main__':
    start_main = time.time()
    main()
    print(f'Worker {WORKER_ID} ran in {time.time() - start_main} seconds')
