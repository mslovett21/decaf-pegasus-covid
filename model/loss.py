import torch.nn.functional as F
from IPython import embed
import torch
def nll_loss(output, target):
    return F.nll_loss(output, target)


def crossentropy_loss(output, target):
    return F.cross_entropy(output, target)

def focal_loss(output,target):
    w2 = torch.Tensor([1,1,4]).cuda()
    loss = F.cross_entropy(output, target,weight=w2)
    y = torch.argmax(F.softmax(output),dim = 1)
    return loss
'''
def focal_loss(output,target):
    w2 = torch.Tensor([1,1,4])

    ce_loss = F.cross_entropy(output, target, reduction='none')
    embed()
    print(output)
    print(target)
    pt = torch.exp(-ce_loss)
    alpha = 0.25
    gamma = 2
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
    return focal_loss
    '''