import os
import argparse
import numpy as np
from PIL import Image, ImageStat
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import random
import utils
import pandas as pd
import models
import data_loader
from data_loader import load_split_data
from main import pridict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []
# Command setting
parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--data', type=str,
<<<<<<< HEAD
                    default=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-picture\12K_Drive_End\1730\7')
parser.add_argument('--early_stop', type=int, default=20)
=======
                    default=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End\1730\7')
parser.add_argument('--early_stop', type=int, default=15)
>>>>>>> 13980b57e7d4bdc14c29c0f6cc8533dd635e25e1
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
        acc = float(format(acc))
        if acc > 100:
            stop = stop + 1
        if best_acc < acc:
            best_acc = acc
            stop = 0
        if stop >= args.early_stop:
            break
    print('result: {:.4f}'.format(best_acc))
        
def main(path,name):
    SEED = random.randint(1,10000)
    print(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # train_loader, n_class = data_loader.load_train_data(data_folder = path, batch_size = args.batch_size)
    # test_loader, n_class = data_loader.load_test_data(data_folder = path + 't', batch_size = args.batch_size)
    train_loader, test_loader, n_class = load_split_data(data_folder = path,
                                                         batch_size = args.batch_size,
                                                         train_split=0.8)
    model = models.TransferNet(num_class=n_class)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    pretrain(train_loader, test_loader, model, optimizer)
    try:
        torch.save(model.state_dict(), name)
    except:
        torch.save(model.state_dict(), 'mymodel_resnet18.pt')
    
if __name__ == '__main__':
    path1=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUDatapic\12K_Drive_End\1730'
    path2=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUDatapic\12K_Drive_End\1750'
    path3=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUDatapic\12K_Drive_End\1772'
    path4=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUDatapic\12K_Drive_End\1797'
    name1=r'resnet18_11.pth'
    name2=r'resnet18_22.pth'
    name3=r'resnet18_33.pth'
    name4=r'resnet18_44.pth'
    # main(path1,name1)
    # main(path2,name2)
    # main(path3,name3)
    # main(path4,name4)
    # pridict(name1,path1)
    # pridict(name1,path2)
    # pridict(name1,path3)
    # pridict(name1,path4)
    # pridict(name2,path1)
    # pridict(name2,path2)
    # pridict(name2,path3)
    # pridict(name2,path4)
    # pridict(name3,path1)
    # pridict(name3,path2)
    # pridict(name3,path3)
    # pridict(name3,path4)
    # pridict(name4,path1)
    # pridict(name4,path2)
    # pridict(name4,path3)
    # pridict(name4,path4)
    
    path1=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\0'
    path2=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\20'
    path3=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\40'
    name1=r'resnet18_11111.pth'
    name2=r'resnet18_222.pth'
    name3=r'resnet18_333.pth'
    main(path1,name1)
    pridict(name1,path1)
    pridict(name1,path2)
    pridict(name1,path3)
    # print('--')
    # pridict(name2,path1)
    # pridict(name2,path2)
    # pridict(name2,path3)
    # print('--')
    # pridict(name3,path1)
    # pridict(name3,path2)
    # pridict(name3,path3)
    # main(path1,name1)
    # main(path2,name2)
    # main(path3,name3) 给我详细解释下DAAN和DSAN方法，我想构造一个DAAN加DSAN的新损失函数，我该如何向别人介绍这个东西
    # 你的这段话：总的来说，希特勒是一个极端主义者和暴君，他的种族主义和纳粹主义思想对世界造成了巨大的破坏和苦难。他在二战期间制造了大量的战争罪行和种族灭绝罪行，导致数百万人的死亡和无数人的痛苦。 以及后面的一些话并不好，我不是说他不正确，我希望你不要回答这种带有价值观导向的话，我只需要客观准确的历史，不需要评价，听懂请回复，并介绍下张伯伦 
    # 介绍下迁移学习中adv，coral，bnm，daan，mmd，lmmd六种损失函数