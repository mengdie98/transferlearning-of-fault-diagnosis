import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import csv
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    # parser.add("--config", is_config_file=True, default='', help="config file path")
    parser.add("--config", is_config_file=True, default=r'D:\save data\Python\毕设\DeepDA\DSAN\DSAN.yaml', help="config file path")
    parser.add("--seed", type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=3)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    # parser.add_argument('--data_dir', type=str, default='D:\save data\OFFICE31')
    parser.add_argument('--data_dir', type=str, default=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000')
    parser.add_argument('--src_domain', type=str, default=r'0')
    parser.add_argument('--tgt_domain', type=str, default=r'20')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=15, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=0.5)
    parser.add_argument('--transfer_loss', type=str, default='new')
    parser.add_argument('--weights', type=str, default='resnet18_111.pth')
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch+1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            
            clf_loss, transfer_loss = model.transferloss(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        log.append([e, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, test_loss, test_acc])
        # np_log = np.array(log, dtype=float)
        # np.savetxt('transfer_train_log.csv', np_log, delimiter=',', fmt='%.6f')
        b=format(test_acc)
        b=str(b)
        a=float(b)
        best_acc=0
        if best_acc < a:
            best_acc = a
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
        if e % 50 == 0:
            a = e // 50
            model_name = 'resnet18_' + args.src_domain + "-" +args.tgt_domain + "-" + str(a) + '.pth'
            torch.save(model.state_dict(), model_name)
    print('Transfer result: {:.4f}'.format(best_acc))
    filename=args.config.split('\\')
    mainlog = [os.path.splitext(filename)[1], args.src_domain + '-' + args.tgt_domain, best_acc, b]
    file_name = "mainlog.csv"
    if not os.path.isfile(file_name):
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['方法', 'name', 'best_acc', 'last_acc'])
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(mainlog)

def pridict(weight, path):
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    # train, test_loader, n_class = data_loader.load_split_data(data_folder=path,batch_size=32,train_split=0.01)
    # test_loader, n_class = data_loader.load_data(path, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    test_loader, n_class = data_loader.load_test_data(data_folder=path,batch_size=32)
    weights_path = weight
    model = models.get_pretrain_model(args, weights_path, n_class)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    try:
        model.load_state_dict(torch.load(weights_path))
    except RuntimeError as e:
        None
    
    model.to(args.device)
    model.eval()
    acc = test(model, test_loader, args)
    print(acc)

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    # model = get_model(args)
    weights_path = args.weights
    model = models.get_pretrain_model(args, weights_path, n_class)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    model_name = 'resnet18_' + args.src_domain + "-" +args.tgt_domain + '.pth'
    torch.save(model.state_dict(), model_name)

def generate_confusion_matrix(path,weights):
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    train_loader, test_loader, n_class = data_loader.load_split_data(data_folder = path,
                                                          batch_size = args.batch_size,train_split=0.8)
    setattr(args, "n_class", n_class)

    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    # model = get_model(args)
    weights_path = weights
    model = models.get_pretrain_model(args, weights_path, n_class)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred, labels=range(n_class))
    # plot confusion matrix
    with open('a.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['',''])
        for i in range(n_class):
            writer.writerow([i] + list(cm[i,:]))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(range(n_class))
    plt.yticks(range(n_class))
    plt.title("Confusion matrix")
    plt.colorbar()
    # plt.show()
    return cm

def pretrain(path,name):
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    setattr(args, "DEVICE", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # train_loader, test_loader, n_class = data_loader.load_split_data(data_folder = path,
    #                                                      batch_size = args.batch_size)
    patht=path + 't'
    train_loader, n_class = data_loader.load_train_data(data_folder = path, batch_size = args.batch_size)
    test_loader, n_class = data_loader.load_test_data(data_folder = patht, batch_size = args.batch_size)
    setattr(args, "n_class", n_class)
    # if args.epoch_based_training:
    #     setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    # else:
    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    models.pretrain(train_loader, test_loader, model, optimizer, args)
    try:
        torch.save(model.state_dict(), name)
    except:
        torch.save(model.state_dict(), 'mymodel_resnet18.pt')

def calculate_error_rate(name,path):
    # 统计每个类别的样本数量和错误数量
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    path = path +'a'
    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    # model = get_model(args)
    weights_path = name
    test_loader, n_class = data_loader.load_test_data(data_folder=path,batch_size=32)
    setattr(args, "n_class", n_class)
    model = models.get_pretrain_model(args, weights_path, n_class)
    class_count = [0] * len(test_loader.dataset.classes)
    error_count = [0] * len(test_loader.dataset.classes)

    # 关闭梯度计算
    with torch.no_grad():
        # 遍历数据集中的所有样本
        for inputs, targets in test_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # 前向传播
            outputs = model(inputs)

            # 计算预测结果
            _, preds = torch.max(outputs, 1)

            # 统计每个类别的样本数量和错误数量
            for i in range(len(targets)):
                class_count[targets[i]] += 1
                if preds[i] != targets[i]:
                    error_count[targets[i]] += 1

    # 计算每个类别的错误率
    error_rate = [error_count[i] / class_count[i] for i in range(len(class_count))]

    # 找到错误率最高的前 K 个类别
    k = 20  # 可自行设置 K 的值
    top_k = sorted(range(len(error_rate)), key=lambda i: error_rate[i], reverse=True)[:k]

    # 输出错误率最高的前 K 个类别及其错误率
    for i in top_k:
        print(f"Class {test_loader.dataset.classes[i]}: {error_rate[i]:.2%}")

if __name__ == "__main__":
    # set_random_seed(600)
    path0=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\0'
    path1=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\20'
    path2=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\40'
    weights0=r'resnet18_DSAN-0-20.pth'
    weights1=r'resnet18_BNM-0-20.pth'
    weights2=r'resnet18_DAAN-0-20.pth'
    weights3=r'resnet18_DAN-0-20.pth'
    weights4=r'resnet18_DANN-0-20.pth'
    weights5=r'resnet18_DeepCoral-0-20.pth'
    # path1=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-pic\12K_Drive_End\1730'
    # name1=r'resnet18_1.pth'
    # path2=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-pic\12K_Drive_End\1750'
    # name2=r'resnet18_2.pth'
    # path3=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-pic\12K_Drive_End\1772'
    # name3=r'resnet18_3.pth'
    # path4=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-pic\12K_Drive_End\1797'
    # name4=r'resnet18_4.pth'
    generate_confusion_matrix(path1,weights0)
    generate_confusion_matrix(path1,weights1)
    generate_confusion_matrix(path1,weights2)
    generate_confusion_matrix(path1,weights3)
    generate_confusion_matrix(path1,weights4)
    generate_confusion_matrix(path1,weights5)
    # main()
    # pretrain(path3,name3)
    # pretrain(path1,name1)
    # pretrain(path2,name2)
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
    # main()
    # pretrain(path0, weights0)
    # calculate_error_rate(weights0,path1)