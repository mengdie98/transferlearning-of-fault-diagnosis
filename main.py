import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    # parser.add("--config", is_config_file=True, default='', help="config file path")
    parser.add("--config", default='DeepDA\DANN\DANN.yaml', help="config file path")
    parser.add("--seed", type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    # parser.add_argument('--data_dir', type=str, default='D:\save data\OFFICE31')
    parser.add_argument('--data_dir', type=str, default=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End')
    parser.add_argument('--src_domain', type=str, default=r'1772\7')
    parser.add_argument('--tgt_domain', type=str, default=r'1730\7')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=250)
    parser.add_argument('--early_stop', type=int, default=25, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss', type=str, default='adv')

    parser.add_argument('--weights', type=str, default='resnet18_2.pth')
    parser.add_argument('--savename', type=str, default='transfer_resnet18_2-0-')
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
        log.append([e, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, test_loss, test_acc])
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('transfer_train_log.csv', np_log, delimiter=',', fmt='%.6f')
        a=format(test_acc)
        a=float(a)
        best_acc=93
        if best_acc > a:
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
        if e % 50 == 0:
            a = e // 50
            model_name = args.savename + str(a) + '.pth'
            torch.save(model.state_dict(), model_name)
    print('Transfer result: {:.4f}'.format(best_acc))

def pridict(name, path):
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    path = path
    train, test_loader, nclass = data_loader.load_split_data(data_folder=path,batch_size=32,train_split=0.01)
    # test_loader, _ = data_loader.load_data(path, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    weights_path = name
    model = models.get_pretrain_model(args, weights_path)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    
    model.to(args.device)
    model.eval()
    acc = test(model, test_loader, args)
    print(acc)

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    # a=random.randint(0,10240)
    # print("seed=",a)
    # set_random_seed(a)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    # model = get_model(args)
    weights_path = args.weights
    model = models.get_pretrain_model(args, weights_path)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    torch.save(model.state_dict(), args.savename + 'LA.pth')
    
def pretrain(path, name):
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    setattr(args, "DEVICE", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    train_loader, test_loader, n_class = data_loader.load_split_data(data_folder = path,
                                                         batch_size = args.batch_size,
                                                         train_split=0.7)
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
        torch.save(model.state_dict(), 'resnet18_2.pt')

if __name__ == "__main__":
    # set_random_seed(2)
    path0=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End\1730\7'
    name0=r'transfer_resnet18_2-0-LA.pth'
    path1=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End\1750\7'
    name1=r'transfer_resnet18_2-1-LA.pth'
    path2=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End\1772\7'
    name2=r'transfer_resnet18_2-3-LA.pth'
    path3=r'D:\data\CWRUData-picture\CWRUData-picture\12K_Drive_End\1797\7'
    name3=r'resnet18_3.pth'
    pridict(name0,path0)
    pridict(name0,path1)
    pridict(name0,path2)
    pridict(name0,path3)
    # main()
    pridict(name1,path0)
    pridict(name1,path1)
    pridict(name1,path2)
    pridict(name1,path3)
    pridict(name2,path0)
    pridict(name2,path1)
    pridict(name2,path2)
    pridict(name2,path3)
    # pridict(name3,path0)
    # pridict(name3,path1)
    # pridict(name3,path2)
    # pridict(name3,path3)
    # main()
    # pretrain(path2,'resnet18_2.pth')
    # pretrain(path1,'resnet18_1.pth')