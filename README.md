# COVIDNet

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


Original work please see: https://github.com/lindawangg/COVID-Net
PyTorch base see: https://github.com/iliasprc/COVIDNet


## Table of Contents

* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Datasets](#datasets)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


<!-- GETTING STARTED -->
## Getting Started

### Installation
To install the required python packages use the following command 
```
pip install -r requirements.txt
```
<!-- USAGE EXAMPLES -->
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
<!-- RESULTS -->
## Results 


implementation of COVID-Net and comparison with CNNs pretrained on ImageNet dataset


### Results in COVIDx  dataset 


| Accuracy (%) | # Params (M) | MACs (G) |        Model        |
|:------------:|:------------:|:--------:|:-------------------:|
|   89.10      |     115.42   |   2.26   |   [COVID-Net-Small] |
|   91.22      |     118.19   |   3.54   |   [COVID-Net-Large](https://drive.google.com/open?id=1-3SKFua_wFl2_aAQMIrj2FhowTX8B551) |
|   94.0       |     -   |   -      |   [Mobilenet V2   ](https://drive.google.com/open?id=19J-1bW6wPl7Kmm0pNagehlM1zk9m37VV) |
|   95.0       |     -   |   -      |   [ResNeXt50-32x4d](https://drive.google.com/open?id=1-BLolPNYMVWSY0Xnm8Y8wjQCapXiPnLx) |
|   94.0       |     -   |   -      | [ResNet-18](https://drive.google.com/open?id=1wxo4gkNGyrhR-1PG8Vr1hj65MfSAHOgJ) |



<!-- Datasets -->
## Datasets


###  COVIDx 

Chest radiography images distribution


|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    8514   |    66    | 16546 |
|  test |   100  |     100   |    10    |   210 |



[contributors-shield]: https://img.shields.io/github/contributors/iliasprc/COVIDNet.svg?style=flat-square
[contributors-url]: https://github.com/iliasprc/COVIDNet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/iliasprc/COVIDNet.svg?style=flat-square
[forks-url]: https://github.com/iliasprc/COVIDNet/network/members

[stars-shield]: https://img.shields.io/github/stars/iliasprc/COVIDNet.svg?style=flat-square
[stars-url]: https://github.com/iliasprc/COVIDNet/stargazers

[issues-shield]: https://img.shields.io/github/issues/iliasprc/COVIDNet.svg?style=flat-square
[issues-url]: https://github.com/iliasprc/COVIDNet/issues
