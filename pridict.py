import os
from PIL import Image
import torch
from data_loader import load_split_data
import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from pretrain import *

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main1(path):

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    # load image
    img_path = path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model = models.TransferNet(num_class=6)

    # load model weights
    weights_path = "D:\save data\Python\mymodel.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model.to(device)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print(output,predict,predict_cla)  
        
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                    predict[i].numpy()))
        plt.show()
        
def main2():
    path = r"E:\毕设论文\CWRU\CWRU_xjs\CWRUData-picture\12K_Drive_End\1730\21"
    train, test_loader, nclass = load_split_data(data_folder=path,batch_size=32,train_split=0.01)
    model = models.TransferNet(num_class=6).to(DEVICE)
    # n_features = model.fc.in_features
    # fc = torch.nn.Linear(n_features, nclass)
    # model.fc = fc
    # load model weights
    weights_path = "D:\save data\Python\mymodel_resnet18.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    
    model.to(device)
    model.eval()
    acc = test(model, test_loader)
    print(acc)
    
    
if __name__ == "__main__":
    # main1()
    main2()