import argparse
import time
import joblib
import numpy as np
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

import utils.util as util
from communication.communication import send_message_to_manager, create_new_trial_object,get_message_from_manager
from trainer.train import initialize, train, validation
from IPython import embed

import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.ERROR)

### ------------------------- LOGGER--------------------------------
logger = logging.getLogger('tunning_log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ARGS = ""
MODEL = 'COVIDNet_small'
STUDY = None
EPOCHS = 10
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = 'logs/'
WORKER_ID = 0 
BATCH_SIZE = 12
TOTAL_TRIALS = 10
EXCHANGE_RATE = 2
OWN_NEW_TRIALS = 0



def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=500, help='steps to print metrics and loss')
    parser.add_argument('--cuda', type=int, default=1, help='use gpu support')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--dataset_name', type=str, default='COVIDx', help='dataset name COVIDx')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet_small',choices=('COVIDNet_small', 'resnet18', 'COVIDNet_large'))
    parser.add_argument('--root_path', type=str, default='./data',help='path to dataset ')
    parser.add_argument('--save', type=str, default='./saved/COVIDNet' + util.datestr(),help='path to checkpoint save directory ')
    parser.add_argument('--epochs', type=int,default=1, help = "number of training epochs")
    parser.add_argument('--trials', type=int, default=2, help = "number of HPO trials")
    parser.add_argument('--worker_id', type=int, default=0, help = "worker id")
    parser.add_argument('--ex_rate',type=int,default=2, help = "info exchange rate in HPO")
    
    args = parser.parse_args()
    
    return args

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
#        if epoch%5 == 0:
#            best_pred_loss = util.save_model(model, optimizer, ARGS, val_metrics, epoch, best_pred_loss, confusion_matrix)
   
    return val_metrics._data.average.recall_mean


def hpo_monitor(study):
    joblib.dump(study,"hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID))

#TODO: check if updates correctly
def hpo_global_update(study, trial):
    
    global OWN_NEW_TRIALS
    OWN_NEW_TRIALS += 1

    if (OWN_NEW_TRIALS == EXCHANGE_RATE):
        send_message_to_manager(study, EXCHANGE_RATE, WORKER_ID)        
        OWN_NEW_TRIALS = 0
        try:
            get_message_from_manager(study,WORKER_ID)
        except Exception as e:
            logger.warning(e)
            logger.warning("No messages from the manager")

    hpo_monitor(study)


def create_study(hpo_checkpoint_file):
   
    global STUDY

    try:
        STUDY = joblib.load("hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID))
        todo_trials = TOTAL_TRIALS - len(STUDY.trials_dataframe())
        print(STUDY.trials_dataframe())
        if todo_trials > 0 :
            logger.info("There are {} trial(s) to do out of {}".format(todo_trials, TOTAL_TRIALS))
            STUDY.optimize(objective, n_trials=todo_trials, timeout=600, callbacks=[hpo_global_update])
    except:
        STUDY = optuna.create_study(direction = 'maximize', study_name = MODEL)
        STUDY.set_user_attr("worker_id", WORKER_ID)
        STUDY.optimize(objective, n_trials=TOTAL_TRIALS, timeout=600, callbacks=[hpo_global_update])



def main():
    
    global MODEL
    global EPOCHS
    global TOTAL_TRIALS
    global WORKER_ID
    global EXCHANGE_RATE
    global ARGS
    
    ARGS = get_arguments()   
    SEED = ARGS.seed
    
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    np.random.seed(SEED)
       
    if (ARGS.cuda):
        torch.cuda.manual_seed(SEED)
    
    EPOCHS        = ARGS.epochs
    TOTAL_TRIALS  = ARGS.trials
    WORKER_ID     = ARGS.worker_id
    EXCHANGE_RATE = ARGS.ex_rate
    MODEL         = ARGS.model

    fh = logging.FileHandler(LOG_DIR + 'main_worker_{}.log'.format(WORKER_ID))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    try:
        hpo_checkpoint_file = "hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID)
        create_study(hpo_checkpoint_file)   
    except Exception as e:
        logger.info(e)    
    finally:
        hpo_monitor(STUDY)

    return 0


if __name__ == '__main__':
    main()
