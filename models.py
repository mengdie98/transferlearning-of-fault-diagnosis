import torch
import torch.nn as nn
# from transfer_losses import TransferLoss
import backbones
import torchvision
from transfer_losses import *
import utils
import pandas as pd
import os

class easyNet(nn.Module):
    def __init__(self, num_class, base_net='dann', **kwargs):
        super(easyNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        feature_dim = self.base_network.output_num()
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.base_network(x)
        x = self.classifier_layer(x)
        return x
    
class resnet18(nn.Module):
    def __init__(self, n_class):
        super(resnet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        n_features = resnet.fc.in_features
        fc = torch.nn.Linear(n_features, n_class)
        self.fc = fc
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return(x)

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet18', transfer_loss='new', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def transferloss(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "new":
            kwargs['source_label'] = source_label
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "new":
            
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def forward(self, x):
        x = self.base_network(x)
        x = self.bottleneck_layer(x)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        if self.transfer_loss == "new":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
       
def get_pretrain_model(args, weights_path, n_class):
    model = TransferNet(
        n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    try:
        model.load_state_dict(torch.load(weights_path))
    except RuntimeError as e:
        error_log_file = "error.txt"  
        if os.path.exists(error_log_file):
            with open(error_log_file, "a") as file:
                file.write(str(e) + "\n")
        else:
            with open(error_log_file, "w") as file:
                file.write(str(e) + "\n")
    return model

def test(model, test_loader, args):
    model.eval().to(args.DEVICE)
    correct = 0
    len_test_dataset = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.DEVICE), target.to(args.DEVICE)
            s_output = model(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_test_dataset
    return acc

log = []
def pretrain(train_loader, test_loader, model, optimizer, args):
    len_train_loader = len(train_loader)
    len_test_loader = len(test_loader)
    best_acc = 0
    stop = 0
    for e in range(args.n_epoch):
        stop += 1
        model.train().to(args.DEVICE)
        n_batch = len_train_loader
        train_loss = utils.AverageMeter()
        iter_train, iter_test = iter(
            train_loader), iter(test_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(len_train_loader):
            data_train, label_train = iter_train.__next__()
            # data_test, label_test = iter_test.__next__()
            data_train, label_train = data_train.to(
                args.DEVICE), label_train.to(args.DEVICE)
            optimizer.zero_grad()
            output = model(data_train)
            loss = criterion(output, label_train)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
        acc = test(model, test_loader, args)
        log.append(
            [e, train_loss.avg, acc.cpu().numpy()])
        pd.DataFrame.from_dict(log).to_csv('train_log.csv', header=[
            'Epoch', 'train_loss', 'acc'])
        print(f'Epoch: [{e:2d}/{args.n_epoch}], train_loss: {train_loss.avg:.4f}, acc: {acc:.4f}')
        acc=format(acc)
        acc=float(acc)
        best_acc=96
        if best_acc > acc:
            stop = 0
        if stop >= args.early_stop:
            break
    print('result: {:.4f}'.format(best_acc))

if __name__ == '__main__':
    model = easyNet(num_class=6)