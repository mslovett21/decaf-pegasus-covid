# a small 2-node example, just a producer and consumer

# --- include the following 4 lines each time ---

import networkx as nx
import os
import imp
wf = imp.load_source('workflow', os.environ['DECAF_PREFIX'] + '/python/decaf.py')

# --- set your options here ---

# path to .so module for dataflow callback functions
mod_path = os.environ['DECAF_PREFIX'] + '/examples/direct/mpmd/mod_dflow.so'

# --- Graph definition ---
prod = wf.Node("prod", start_proc=0, nprocs=1, func='prod', cmdline='python3 ./worker.py --epochs 1 --ex_rate 2 --cuda 0') #epoch exchange_rate (for first_iter)
inPort = prod.addInputPort("in")
inPort.setTokens(1)
outPort = prod.addOutputPort("out")

con = wf.Node("con", start_proc=1, nprocs=1, func='con', cmdline='python3 ./master.py 4 1 4') #total_trials exchange_rate cp_freq
inPort2 = con.addInputPort("in")
outPort2 = con.addOutputPort("out")


linkPC = wf.Edge(prod.getOutputPort("out"), con.getInputPort("in"), start_proc=0, nprocs=0, func='dflow',
        path=mod_path, prod_dflow_redist='count', dflow_con_redist='count', cmdline='./dflow')

linkCP = wf.Edge(con.getOutputPort("out"), prod.getInputPort("in"), start_proc=0, nprocs=0, func='dflow',
        path=mod_path, prod_dflow_redist='proc', dflow_con_redist='proc', cmdline='./dflow')

# --- convert the nx graph into a workflow data structure and run the workflow ---
wf.processGraph("optuna")
