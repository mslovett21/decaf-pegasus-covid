# COVIDNet


Original work: https://github.com/lindawangg/COVID-Net <br>
PyTorch base see: https://github.com/iliasprc/COVIDNet


## Usage

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
