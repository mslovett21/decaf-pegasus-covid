# COVIDNet

Publication: https://arxiv.org/pdf/2003.09871.pdf <br>
Original work: https://github.com/lindawangg/COVID-Net <br>
PyTorch base see: https://github.com/iliasprc/COVIDNet


## TEST COMMUNICATION
```
python main.py --worker_id 0 --epochs 1 --trials 2
python main.py --worker_id 0 --epochs 1 --trials 2
```
These runs produce the following two text files: <br>
<br>
**worker_id_0_to_manager.txt** :

```
{'optimizer': 'SGD', 'weight_decay': 0.020892347311734047, 'learning_rate': 3.4722760860520766e-07, 'worker_id': 0, 'value': 0.07482993197278912}
{'optimizer': 'RMSprop', 'weight_decay': 0.0001983594447281459, 'learning_rate': 1.329122307006713e-06, 'worker_id': 0, 'value': 0.1428571428571428}
```
and **worker_id_1_to_manager.txt**:
```
{'optimizer': 'Adam', 'weight_decay': 0.00030238417060408793, 'learning_rate': 3.7617520347549393e-06, 'worker_id': 1, 'value': 0.1428571428571428}
{'optimizer': 'RMSprop', 'weight_decay': 0.0560467627843438, 'learning_rate': 7.643195253422718e-06, 'worker_id': 1, 'value': 0.13605442176870744}
```

Then run **manager.py** script
```
python manager.py 0
```

that creates **manager_memory.csv** (when it runs first time). <br>
It adds trials from **worker_id_0_to_manager.txt** to **manager_memory.csv**

```
optimizer,weight_decay,learning_rate,worker_id,value
SGD,0.020892347311734047,3.4722760860520766e-07,0,0.07482993197278912
RMSprop,0.0001983594447281459,1.329122307006713e-06,0,0.1428571428571428
```
then run

```
python manager.py 1
```
This will add trials from **worker_id_1_to_manager.txt** to **manager_memory.csv**  <br>
<br>
Additionally it creates a file called:

```
manager_to_worker_id_1.txt
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
