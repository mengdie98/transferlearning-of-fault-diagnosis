import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

# 定义数据增强和预处理的操作
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = datasets.ImageFolder('path/to/data', transform=transform)

# 计算每个类别的权重
num_samples = len(dataset)
num_classes = len(dataset.classes)
class_count = [0] * num_classes
for _, label in dataset:
    class_count[label] += 1
class_weight = [num_samples / (num_classes * count) for count in class_count]

# 定义采样器
sampler = WeightedRandomSampler(class_weight, num_samples)

# 定义数据加载器
batch_size = 32
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
