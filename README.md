# COVIDNet

Publication: https://arxiv.org/pdf/2003.09871.pdf <br>
Original work: https://github.com/lindawangg/COVID-Net <br>
PyTorch base see: https://github.com/iliasprc/COVIDNet


## REQUIREMENTS
```
requirements.txt  --- the minimum needed
requirements2.txt --- my whole environment
```
## TEST COMMUNICATION
```
python main.py --worker_id 0 --epochs 1 --trials 2
python main.py --worker_id 0 --epochs 1 --trials 2
```
These runs produce the following two text files: <br>
<br>
**worker_id_0_to_manager.txt**

```
{'optimizer': 'RMSprop', 'weight_decay': 0.0029527134878696458, 'learning_rate': 1.480510371973113e-06, 'worker_id': 0, 'value': 0.1428571428571428}
{'optimizer': 'SGD', 'weight_decay': 0.012962593884789587, 'learning_rate': 2.6737290192971144e-06, 'worker_id': 0, 'value': 0.1428571428571428}
```
and **worker_id_1_to_manager.txt**
```
{'optimizer': 'SGD', 'weight_decay': 0.02227013100110008, 'learning_rate': 1.954959853042296e-06, 'worker_id': 1, 'value': 0.07482993197278912}
{'optimizer': 'Adam', 'weight_decay': 0.015326215774491649, 'learning_rate': 7.523176817240329e-07, 'worker_id': 1, 'value': 0.1428571428571428}
```

Then run **manager.py** script
```
python manager.py 0
```

that creates **manager_memory.csv** (when it runs first time). <br>
<br>
It adds trials from **worker_id_0_to_manager.txt** to **manager_memory.csv**

```
optimizer,weight_decay,learning_rate,worker_id,value
RMSprop,0.0029527134878696458,1.480510371973113e-06,0,0.1428571428571428
SGD,0.012962593884789587,2.6737290192971144e-06,0,0.1428571428571428
```
then run

```
python manager.py 1
```
This will add trials from **worker_id_1_to_manager.txt** to **manager_memory.csv**  <br>
<br>
Additionally it creates a file called **manager_to_worker_id_1.txt** with the trials from manager that
were added by other workers since the last communication between the worker 1 and manager.

```
{'optimizer': 'RMSprop', 'weight_decay': 0.0029527134878696453, 'learning_rate': 1.480510371973113e-06, 'worker_id': 0, 'value': 0.1428571428571428}
{'optimizer': 'SGD', 'weight_decay': 0.012962593884789587, 'learning_rate': 2.6737290192971144e-06, 'worker_id': 0, 'value': 0.1428571428571428}

```
Then if you run:
```
python main.py --worker_id 1 --epochs 1 --trials 6

```
At the end of the execution of the command - the STUDY object will contain results of 8 trials:<br>
2 trials from previous execution,then the script will decide to execute 4 more trials. <br>
Since the exchange rate is 2.<br>
After executing 2 more trials the communication will be performed and the worker will send the 2 new results to the manager and it will recive 2 trials executed by worker with ID 0 from the manager. <br>
After adding them, it will continue with executing two more trials. Result is 8, 6 trials of its own and another 2 from the other worker.
```
   number     value             datetime_start  ... params_weight_decay user_attrs_worker_id     state
0       0  0.074830 2020-09-06 21:31:34.036646  ...            0.022270                    1  COMPLETE
1       1  0.142857 2020-09-06 21:32:17.178423  ...            0.015326                    1  COMPLETE
2       2  0.142857 2020-09-06 21:34:59.284312  ...            0.013500                    1  COMPLETE
3       3  0.142857 2020-09-06 21:35:46.885240  ...            0.000815                    1  COMPLETE
4       4  0.142857 2020-09-06 21:36:30.528158  ...            0.002953                    0  COMPLETE
5       5  0.142857 2020-09-06 21:36:30.528414  ...            0.012963                    0  COMPLETE
6       6  0.142857 2020-09-06 21:36:30.531897  ...            0.000027                    1  COMPLETE
7       7  0.142857 2020-09-06 21:37:17.239606  ...            0.007243                    1  COMPLETE
8       8  0.142857 2020-09-06 21:38:01.420274  ...            0.002953                    0  COMPLETE
9       9  0.142857 2020-09-06 21:38:01.420485  ...            0.012963                    0  COMPLETE

```

### Training

The network takes as input an image of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, C), where N is the number of batches and C number of output classes.

1. To train the Network from scratch simply do `python main.py` 
 Arguments for training 
 ```
   -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size for training
  --log_interval LOG_INTERVAL
                        steps to print metrics and loss
  --dataset_name DATASET_NAME
                        dataset name
  --nEpochs NEPOCHS     total number of epochs
  --device DEVICE       gpu device
  --seed SEED           select seed number for reproducibility
  --classes CLASSES     dataset classes
  --lr LR               learning rate (default: 1e-3)
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 1e-6)
  --cuda                use gpu for speed-up
  --tensorboard         use tensorboard for loggging and visualization
  --resume PATH         path to latest checkpoint (default: none)
  --model {COVIDNet_small,resnet18,mobilenet_v2,densenet169,COVIDNet_large}
  --opt {sgd,adam,rmsprop}
  --root_path ROOT_PATH
                        path to dataset
  --save SAVE           path to checkpoint save directory


```

## DATASET


###  COVIDx 



|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    8514   |    66    | 16546 |
|  test |   100  |     100   |    10    |   210 |
