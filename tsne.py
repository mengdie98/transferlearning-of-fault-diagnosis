import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import models
import configargparse
from utils import *

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    # parser.add("--config", is_config_file=True, default='', help="config file path")
    parser.add("--config", default=r'D:\save data\Python\毕设\DeepDA\DSAN\DSAN.yaml', help="config file path")
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
    parser.add_argument('--transfer_loss', type=str, default='')
    parser.add_argument('--weights', type=str, default='resnet18_111.pth')
    return parser
import random

# 数据集路径
parser = get_parser()
args = parser.parse_args()
setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
data_path = r"E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data-pic\with_box\1000\20"
data_path2 = r"E:\毕设论文\CWRU\CWRU_xjs\CWRUData-pic\12K_Drive_End\1750"
# 定义数据转换，可以根据您的需求进行修改
transform = transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.26, 0.17, 0.45],
                                  std=[0.03, 0.11, 0.05])])

# 加载数据集
dataset = ImageFolder(data_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
dataset2 = ImageFolder(data_path2, transform=transform)
data_loader2 = DataLoader(dataset2, batch_size=32, shuffle=False)
# 提取特征向量
weights="resnet18_DeepCoral-0-20.pth"
# model = torch.load("resnet18_1.pth")

model = models.get_pretrain_model(args,weights,6)
# model= models.resnet18(6)
# try:
#     model = torch.load("resnet18_1.pth")
# except Exception as e:
#     print(e)
a=555
model.eval()
features = []
with torch.no_grad():
    for images, _ in data_loader:
        images=images.to(args.device)
        feats = model(images)
        feats = feats.cpu().numpy()
        features.append(feats)
    # for images, _ in data_loader2:
    #     images=images.to(args.device)
    #     feats = model(images)
    #     feats = feats.cpu().numpy()
    #     features.append(feats)
features = np.concatenate(features)
# print(features)
# 对特征向量进行 T-SNE 降维
tsne = TSNE(n_components=2, init='pca', random_state=0)
# , random_state=0
tsne_results = tsne.fit_transform(features)

# 可视化
labels = dataset.targets
classes = dataset.classes
classes = ['ball','inner_ring','normal','outer_ring_3','outer_ring_6','outer_ring_12s']
num_classes = len(classes)

markers = ['o', '^']
fig, ax = plt.subplots()
color = ['red','green','blue','brown','black','yellow']
# 'pink','olive','navy','peach','silver','gold'
for i in range(6):
    if i < 6:
        ax.scatter(tsne_results[a*i:a*(i+1)-int(a/2), 0], tsne_results[a*i:a*(i+1)-int(a/2), 1], label=classes[i],s=1, marker='o',c=color[i])
    else:
        # ax.scatter(tsne_results[a*i:a*(i+1)-int(a/2), 0], tsne_results[a*i:a*(i+1)-int(a/2), 1], s=1, marker='^', c=color[i-6])
        pass
# ax.legend()
ax.legend(loc='upper left', bbox_to_anchor=(0.65, 0.95), fontsize=8)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.show()
plt.show()
filename = weights+'.png'
# plt.savefig(filename)
plt.clf()
plt.close(fig)

# plt.text(20, 1, 'aaa')
# plt.scatter(s=50)怎么把生成的图片中的label变小点或者移动到图片外，他已经妨碍了一些点的显示了
