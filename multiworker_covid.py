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
worker1 = wf.Node("worker1", start_proc=0, nprocs=1, func='prod', cmdline='python ./worker.py --ex_rate 1') #epoch exchange_rate (for first_iter)
worker1InPort = worker1.addInputPort("in")
worker1InPort.setTokens(1)
worker1OutPort = worker1.addOutputPort("out")

master = wf.Node("master", start_proc=1, nprocs=1, func='con', cmdline='python ./master.py 3 1 8') #total_trials exchange_rate cp_freq
masterInPort = master.addInputPort("in")
masterOutPort = master.addOutputPort("out")

worker2 = wf.Node("worker2", start_proc=2, nprocs=1, func='prod', cmdline='python ./worker.py --ex_rate 1') #epoch exchange_rate (for first_iter)
worker2InPort = worker2.addInputPort("in")
worker2InPort.setTokens(1)
worker2OutPort = worker2.addOutputPort("out")

linkW1M = wf.Edge(worker1.getOutputPort("out"), master.getInputPort("in"), start_proc=0, nprocs=0, func='dflow', path=mod_path, prod_dflow_redist='count', dflow_con_redist='count', cmdline='./dflow')

linkW2M = wf.Edge(worker2.getOutputPort("out"), master.getInputPort("in"), start_proc=0, nprocs=0, func='dflow', path=mod_path, prod_dflow_redist='count', dflow_con_redist='count', cmdline='./dflow')

linkMW1 = wf.Edge(master.getOutputPort("out"), worker1.getInputPort("in"), start_proc=0, nprocs=0, func='dflow', path=mod_path, prod_dflow_redist='proc', dflow_con_redist='proc', cmdline='./dflow')

linkMW2 = wf.Edge(master.getOutputPort("out"), worker2.getInputPort("in"), start_proc=0, nprocs=0, func='dflow', path=mod_path, prod_dflow_redist='proc', dflow_con_redist='proc', cmdline='./dflow')

# --- convert the nx graph into a workflow data structure and run the workflow ---
wf.processGraph("optuna")
