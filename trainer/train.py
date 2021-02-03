import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_loader.covidxdataset import COVIDxDataset
from model.metric import accuracy, precision_score
from model.loss import weighted_loss
from utils.util import print_stats, print_summary, select_model, MetricTracker


import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution,LogUniformDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)
from IPython import embed


METRICS_TRACKED = ['loss', 'correct', 'total', 'accuracy','precision_mean', 'recall_mean']

def initialize(args):
    
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    model = select_model(args)

    if (args.cuda):
        model.cuda()

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 2}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 1}

    train_loader = COVIDxDataset(mode='train', n_classes=args.classes)
    val_loader   = COVIDxDataset(mode='test', n_classes=args.classes)
    test_loader  = None

    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    test_generator = None


    return model, training_generator, val_generator, test_generator


def train(args, model, trainloader, optimizer, epoch):
    
    start_time = time.time()
    model.train()

    train_metrics = MetricTracker(*[m for m in METRICS_TRACKED], mode='train')
    w2 = torch.Tensor([1.0,1.0,1.5])
    
    if (args.cuda):
        print("CUDA true")
        model.cuda()
        w2 = w2.cuda()
    
    train_metrics.reset()
    # JUST FOR CHECK
    counter_batches = 0
    counter_covid = 0

    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        counter_batches +=1

        if (args.cuda):
            input_data = input_data.cuda()
            target = target.cuda()

        output = model(input_data)

        loss,counter = weighted_loss(output, target,w2)
        counter_covid += counter
        loss.backward()

        optimizer.step()
        correct, total, acc = accuracy(output, target)
        precision_mean, recall_mean = precision_score(output,target)

        num_samples = batch_idx * args.batch_size + 1
        train_metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(),
         'accuracy': acc, 'precision_mean': precision_mean, 'recall_mean':recall_mean},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        print_stats(args, epoch, num_samples, trainloader, train_metrics)
    print("--- %s seconds ---" % (time.time() - start_time))
    print_summary(args, epoch, num_samples, train_metrics, mode="Training")
    return train_metrics


def validation(args, model, testloader, epoch):
    
    model.eval()
    
    val_metrics = MetricTracker(*[m for m in METRICS_TRACKED], mode='val')
    val_metrics.reset()
    w2 = torch.Tensor([1.0,1.0,1.5]) #w_full = torch.Tensor([1.456,1.0,15.71])
    
    if (args.cuda):
        w2 = w2.cuda()
    
    confusion_matrix = torch.zeros(args.classes, args.classes)
    
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors

            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()

            output = model(input_data)

            loss,counter = weighted_loss(output, target,w2)
            correct, total, acc = accuracy(output, target)
            precision_mean, recall_mean = precision_score(output,target)
            
            num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(output, 1)
            
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            val_metrics.update_all_metrics({'correct': correct, 'total': total, 'loss': loss.item(),
             'accuracy': acc,'precision_mean': precision_mean, 'recall_mean':recall_mean},
                                           writer_step=(epoch - 1) * len(testloader) + batch_idx)


    print_summary(args, epoch, num_samples, val_metrics, mode="Validation")
    print('Confusion Matrix\n {}'.format(confusion_matrix.cpu().numpy()))
    
    return val_metrics, confusion_matrix
