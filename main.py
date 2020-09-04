import argparse
import time
import joblib
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


import utils.util as util
from communication.communication import send_message_to_manager, create_new_trial_object
from trainer.train import initialize, train, validation
from IPython import embed


import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL      = 'COVIDNet_small'
EPOCHS     = 10
BATCH_SIZE = 12
WORKER_ID  = 0 
STUDY      = None
TOTAL_TRIALS  = 10
EXCHANGE_RATE = 2
OWN_NEW_TRIALS    = 0
ARGS = ""


def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=500, help='steps to print metrics and loss')
    parser.add_argument('--dataset_name', type=str, default='COVIDx', help='dataset name COVIDx')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--cuda', action='store_true', default=True, help='use gpu for speed-up')
    parser.add_argument('--tensorboard', action='store_true', default=True,help='use tensorboard for loggging and visualization')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet_small',choices=('COVIDNet_small', 'resnet18', 'COVIDNet_large'))
    parser.add_argument('--root_path', type=str, default='./data',help='path to dataset ')
    parser.add_argument('--save', type=str, default='./saved/COVIDNet' + util.datestr(),help='path to checkpoint save directory ')
    parser.add_argument('--epochs',  metavar='num_epochs', type=int, nargs=1,default=1, help = "number of training epochs")
    parser.add_argument('--trials',  metavar='num_trials', type=int, nargs=1, default=1, help = "number of HPO trials")
    parser.add_argument('--worker_id', metavar='worker_id', type=int, nargs=1,  default=0, help = "worker id")
    parser.add_argument('--ex_rate',  metavar='exchange_rate',type=int, nargs=1,  default=2, help = "info exchange rate in HPO")
    
    args = parser.parse_args()
    
    return args

def objective(trial):

    model, training_generator, val_generator, test_generator = initialize(ARGS)

    optim_name   = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    lr           = trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True)
    
    optimizer = util.select_optimizer(optim_name, model, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5, verbose=True)

    best_pred_loss = 1000.0

    if ARGS.tensorboard:
        writer = SummaryWriter('./runs/' + util.datestr())
    else:
        writer = None
        
    for epoch in range(1, EPOCHS + 1):
        
        train(ARGS, model, training_generator, optimizer, epoch, writer)        
        val_metrics, confusion_matrix = validation(ARGS, model, val_generator, epoch, writer)
        
        best_pred_loss = util.save_model(model, optimizer, ARGS, val_metrics, epoch, best_pred_loss, confusion_matrix)
        scheduler.step(val_metrics._data.average.loss)
        
        print("Sensitivity: {:.4f}".format(val_metrics._data.average.recall_mean))
    
    return val_metrics._data.average.recall_mean




def hpo_monitor(study):
    joblib.dump(study,"hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID))



def hpo_global_update(study, trial):
    
    global OWN_NEW_TRIALS
    OWN_NEW_TRIALS += 1

    if (OWN_NEW_TRIALS == EXCHANGE_RATE):
        send_message_to_manager(study, EXCHANGE_RATE, WORKER_ID)        
        OWN_NEW_TRIALS = 0
        try:
            get_message_from_manager(study)
        except Exception as e:
            print(e)
            print("No messages from the manager")


def create_study(hpo_checkpoint_file):
   
    global STUDY

    try:
        STUDY = joblib.load("hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID))
        todo_trials = TOTAL_TRIALS - len(STUDY.trials_dataframe())
        print(STUDY.trials_dataframe())
        if todo_trials > 0 :
            print("There are {} trial(s) to do out of {}".format(todo_trials, TOTAL_TRIALS))
            STUDY.optimize(objective, n_trials=todo_trials, timeout=600, callbacks=[hpo_global_update])
        else:
            pass
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

    
    EPOCHS        = ARGS.epochs[0]
    TOTAL_TRIALS  = ARGS.trials[0]
    WORKER_ID     = ARGS.worker_id[0]
    EXCHANGE_RATE = ARGS.ex_rate
    MODEL         = ARGS.model
    
    try:
        hpo_checkpoint_file = "hpo_study_checkpoint_{}_{}.pkl".format(MODEL, WORKER_ID)
        create_study(hpo_checkpoint_file)
    
    except Exception as e:
        print(e)
    
    finally:
        hpo_monitor(STUDY)

    return 0




if __name__ == '__main__':
    main()
