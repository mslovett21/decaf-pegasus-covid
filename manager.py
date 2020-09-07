#!/usr/bin/env python3
import argparse
import logging
import os,sys
import pandas as pd
from IPython import embed



### ------------------------- LOGGER--------------------------------
logger = logging.getLogger('manager_log')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


### ------------------------- PARSER --------------------------------
parser = argparse.ArgumentParser(description='Manager for sharing HPO results between workers')
parser.add_argument('worker_id',  metavar='worker_id', type=int, nargs=1, help = "worker id")

MANAGER_MEMORY_FILE = "manager_memory.csv"




def readin_workers_data(f_receive, FIRST_COMM_FLAG, MANAGER_MEMORY):
    
    if FIRST_COMM_FLAG:
        data = eval(f_receive.readline())
        MANAGER_MEMORY = pd.DataFrame([data], columns=data.keys())
    
    trials = f_receive.readlines()
    f_receive.close()
    
    for trial in trials:
        MANAGER_MEMORY  = MANAGER_MEMORY.append(eval(trial), ignore_index=True)

    return MANAGER_MEMORY


def prepare_message_for_worker(MANAGER_MEMORY, WORKER_ID):
    
    if MANAGER_MEMORY is None:
        return None

    try:
        last_entry_index = MANAGER_MEMORY[MANAGER_MEMORY.worker_id==WORKER_ID].index[-1]
    except Exception as e:
        logger.info("No previous trials recorded from this worker ")
        return MANAGER_MEMORY

    try:
        return MANAGER_MEMORY[last_entry_index+1:]
    except Exception as e:
        return None


def save_worker_message(WORKER_ID, message):
    f_send = open("manager_to_worker_id_{}.txt".format(WORKER_ID),"w+")
    row_dict = message.to_dict(orient="records")
    for r in row_dict:
        f_send.write(str(r) + "\n")


def save_manager_memory(MANAGER_MEMORY):
    MANAGER_MEMORY.to_csv(MANAGER_MEMORY_FILE, index=False)



### -------------------------MAIN--------------------------------
def main():

    WORKER_ID = 0 
    MANAGER_MEMORY = None
    FIRST_COMM_FLAG = False

    args = parser.parse_args()
    
    WORKER_ID = args.worker_id[0]
    FIRST_COMM_FLAG = False
    
    file_from_worker = "worker_id_{}_to_manager.txt".format(WORKER_ID)

    try:
        MANAGER_MEMORY = pd.read_csv(MANAGER_MEMORY_FILE)
    except Exception as e:
        FIRST_COMM_FLAG = True
        logger.info("No previous memory file available.")
        pass
    
    try:
        f_receive = open(file_from_worker.format(WORKER_ID),"r")
    except Exception as e:
        logger.error("File with a message from worker does not exist.")
        return -1

    message_for_worker = prepare_message_for_worker(MANAGER_MEMORY, WORKER_ID)
    
    if message_for_worker is None:
        logger.info("No message for the worker with ID: {}".format(WORKER_ID))
    else:
        save_worker_message(WORKER_ID, message_for_worker)

    MANAGER_MEMORY = readin_workers_data(f_receive, FIRST_COMM_FLAG, MANAGER_MEMORY)
    save_manager_memory(MANAGER_MEMORY)

    return 0



if __name__ == '__main__':
    main()