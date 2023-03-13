import os
import argparse
import numpy as np
from PIL import Image, ImageStat
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from data_loader import load_split_data
import random
import utils
import pandas as pd
import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []
# Command setting
parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--data', type=str,
                    default=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End\1730\7')
parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--lamb', type=float, default=.01)
args = parser.parse_args()

def test(model, test_loader):
    model.eval().to(DEVICE)
    correct = 0
    len_test_dataset = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_test_dataset
    return acc

def pretrain(train_loader, test_loader, model, optimizer):
    len_train_loader = len(train_loader)
    len_test_loader = len(test_loader)
    best_acc = 0
    stop = 0
    for e in range(args.n_epoch):
        stop += 1
        model.train().to(DEVICE)
        n_batch = len_train_loader
        train_loss = utils.AverageMeter()
        iter_train, iter_test = iter(
            train_loader), iter(test_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(len_train_loader):
            data_train, label_train = iter_train.__next__()
            # data_test, label_test = iter_test.__next__()
            data_train, label_train = data_train.to(
                DEVICE), label_train.to(DEVICE)
            optimizer.zero_grad()
            output = model(data_train)
            loss = criterion(output, label_train)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
        acc = test(model, test_loader)
        log.append(
            [e, train_loss.avg, acc.cpu().numpy()])
        pd.DataFrame.from_dict(log).to_csv('train_log.csv', header=[
            'Epoch', 'train_loss', 'acc'])
        print(f'Epoch: [{e:2d}/{args.n_epoch}], train_loss: {train_loss.avg:.4f}, acc: {acc:.4f}')
        if best_acc < acc:
            best_acc = acc
            stop = 0
        if stop >= args.early_stop:
            break
    print('result: {:.4f}'.format(best_acc))
        

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_loader, test_loader, n_class = load_split_data(data_folder = args.data,
                                                         batch_size = args.batchsize,
                                                         train_split=0.7)
    # model = torchvision.models.resnet34(pretrained=False).to(DEVICE)
    # n_features = model.fc.in_features
    # fc = torch.nn.Linear(n_features, n_class)
    # model.fc = fc
    model = models.TransferNet(num_class=n_class)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    pretrain(train_loader, test_loader, model, optimizer)
    try:
        torch.save(model.state_dict(), 'mymodel_resnet18.pth')
    except:
        torch.save(model.state_dict(), 'mymodel_resnet18.pt')
    