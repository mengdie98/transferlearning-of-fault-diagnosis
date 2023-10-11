import os
import numpy as np
from PIL import Image, ImageStat
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms,datasets
from torch.utils.data.sampler import WeightedRandomSampler
# from sklearn.model_selection import train_test_split
import json
from torch.utils.data.sampler import WeightedRandomSampler,Sampler
import random
from collections import defaultdict

def get_dataset_statistics(path, image_size=(300, 300)):
    """
    计算给定数据集的图像数据均值和标准差
    Args:
        root_dir: 数据集根目录，包含多个文件夹，每个文件夹中包含多张图片

    Returns:
        数据集图像数据的均值和标准差，形如((R_mean, G_mean, B_mean), (R_std, G_std, B_std))
    """
    img_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img = Image.open(os.path.join(root, file)).convert('RGB')
                img = img.resize(image_size)
                img_list.append(np.array(img))

    img_array = np.array(img_list)
    mean = np.mean(img_array, axis=(0, 1, 2))
    std = np.std(img_array, axis=(0, 1, 2))
    return mean, std


def load_split_data(data_folder, batch_size, train_split, num_workers=4, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([random.randint(224,256), random.randint(224,256)]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.26, 0.17, 0.45],
                                  std=[0.03, 0.11, 0.05])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.26, 0.17, 0.45],
                                  std=[0.03, 0.11, 0.05])])
    }
    data = datasets.ImageFolder(root=data_folder, transform = transform['train'])
    flower_list = data.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    train_size = int(train_split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    # 为测试集设置新的transform
    train_dataset.transform = transform['train']
    test_dataset.transform = transform['test']
    
    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    # data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True if train else False, num_workers=num_workers, **kwargs)
    n_class = len(data.classes)
    return train_loader, test_loader, n_class

def load_test_data(data_folder, batch_size, num_workers=4, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.26, 0.17, 0.45],
                                  std=[0.03, 0.11, 0.05])])
    }
    data = datasets.ImageFolder(root=data_folder, transform = transform['test'])
    n_class = len(data.classes)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    return test_loader, n_class

def load_train_data(data_folder, batch_size, num_workers=4, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([random.randint(224,256), random.randint(224,256)]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])])
    }
    data = datasets.ImageFolder(root=data_folder, transform = transform['train'])
    n_class = len(data.classes)
    data.transform = transform['test']
    sampler=MYSampler(data,max_samples=454)
    test_loader = DataLoader(data, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=num_workers)
    return test_loader, n_class
    
def load_data(data_folder, batch_size, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([random.randint(224,256), random.randint(224,256)]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class


def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    sampler=MYSampler(dataset,max_samples=454)
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=sampler if shuffle==True else None, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0

class MYSampler(Sampler):
    def __init__(self, dataset, max_samples=512):
        self.dataset = dataset
        self.max_samples = max_samples
        self.label_indices = defaultdict(list)


        # 根据标签将所有样本的索引分组
        for idx, (data, label) in enumerate(dataset):
            self.label_indices[label].append(idx)

    def __iter__(self):
        indices = []

        # 遍历每个标签
        for label, indices_for_label in self.label_indices.items():
            # 如果标签下的样本数量小于等于max_samples，直接将所有样本的索引添加到indices列表中
            if len(indices_for_label) <= self.max_samples:
                indices += indices_for_label
            # 如果标签下的样本数量大于max_samples，随机选择max_samples个样本的索引
            else:
                indices += random.sample(indices_for_label, self.max_samples)

        # 打乱所有样本的索引顺序，并返回迭代器
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        # 计算所有标签下的样本数量之和
        self.sum=0
        for label, indices_for_label in self.label_indices.items():
            # 如果标签下的样本数量小于等于max_samples，直接将所有样本的索引添加到indices列表中
            if len(indices_for_label) <= self.max_samples:
                self.sum += len(indices_for_label)
            # 如果标签下的样本数量大于max_samples，随机选择max_samples个样本的索引
            else:
                self.sum += self.max_samples

        return self.sum

    
if __name__ == '__main__':
    # train_loader, test_loader, n_class = load_split_data(data_folder=r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-picture\12K_Drive_End\1730\7', batch_size=32, train_split=0.7)
    # print(n_class,'/n',train_loader, test_loader)
# train_dataset = CustomDataset('/path/to/dataset', transform=train_transform)
    print(get_dataset_statistics(path=r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\0'))
# 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # 获取图像数据的均值和标准差
# data_mean, data_std = get_dataset_statistics(r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-picture\12K_Drive_End\1730\7')
# print(data_mean, data_std)
# # 定义数据预处理方法
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(data_mean, data_std)
# ])
