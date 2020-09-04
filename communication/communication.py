import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)

from IPython import embed

def prepare_message(list_of_trials, worker_id):

    message_to_pass = []
    for trial in list_of_trials:
        trial_info = trial.params
        trial_info["worker_id"] = worker_id
        trial_info["value"] = trial.value
        message_to_pass.append(trial_info)
    return message_to_pass



def send_message_to_manager(study,exchange_rate,worker_id):

    message_to_send = prepare_message(study.trials[-exchange_rate:], worker_id)
    f_send = open("worker_id{}_to_manager_{}.txt".format(worker_id, len(study.trials)),"w")
    print(f_send)
    for line in message_to_send:
        f_send.write( str(line))
        f_send.write("\n")
    
    f_send.close()


def get_message_from_manager(study):

    f_receive = open("manager_to_worker_id{}.txt".format(WORKER_ID))
    info_line = f_receive.readline()
    while info_line:
        info_dict = eval(info_line)
        new_trial = create_new_trial_object(info_dict)
        study.add_trial(new_trial)
        print("New trial was added")
        info_line = f_receive.readline()


def create_new_trial_object(trial_info_dict):
    embed()
    
    new_trial = optuna.trial.create_trial(
        params = {"dropout": trial_info_dict["dropout"] ,"optimizer": trial_info_dict["optimizer"], "lr":trial_info_dict["lr"] },
        distributions = {"dropout": UniformDistribution(0.2, 0.5),
        "optimizer" : CategoricalDistribution(choices=('Adam', 'RMSprop')),
        "lr": LogUniformDistribution(1e-05,0.1)},
        value= trial_info_dict["value"],)
    #new_trial.user_attr = ("worker_id",trial_info_dict["worker_id"])
    
    return new_trial