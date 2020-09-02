import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from IPython import embed

#target_names = ['pneumonia', 'normal', 'COVID-19']
#report = classification_report(target.cpu().numpy(), y.cpu().numpy(), target_names=target_names, output_dict=True)

def precision_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        y = pred
        assert pred.shape[0] == len(target)
        report = precision_recall_fscore_support(target.cpu(), y.cpu(), average=None,labels=[0,1,2])
        mean_precision = report[0].mean()
        mean_recall = report[1].mean()
    return mean_precision, mean_recall

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        y = pred
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target), correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
