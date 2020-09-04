#!/usr/bin/env python3


import argparse
import os,sys


WORKER_ID = 0


### ------------------------- PARSER --------------------------------
parser = argparse.ArgumentParser(description='Manager for sharing HPO results between workers')
parser.add_argument('worker_id',  metavar='worker_id', type=int, nargs=1, help = "worker id")



### -------------------------MAIN--------------------------------
def main():


    global WORKER_ID


    args          = parser.parse_args()
    WORKER_ID     = args.worker_id[0]
    print("Interacting with worker {}".format(WORKER_ID))
    # read in your memory if you have any
    # fetch all the trials info that were added since the worker communicated last time
    # add info about trials brought by worker_id from ("worker_id{}_to_manager_{}.txt")
    # send the fetched trials info to the worker ("manager_to_worker_id{}.txt")


    
    #f_receive = open("manager_to_worker_id{}.txt".format(WORKER_ID))
    #f_send = open("worker_id{}_to_manager_{}.txt".format(WORKER_ID, len(study.trials)),"w")





    except Exception as e:
    	print(e)
    finally:
        pass
    return 0



if __name__ == '__main__':
    main()