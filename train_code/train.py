
"""
作者：yu
特别说明：此项目的数据集制作比较麻烦，且很难获取大量训练样本，
因此所有训练样本来源于程序模拟，有需要的话联系qq 2840993104 远程指导

"""


import torch
from torchvision.models import resnet34,resnet18,resnet50
import numpy as np
import os
import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  # 画图工具包 matplotlib
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models

save_dir = "../train_logs"
os.makedirs(save_dir,exist_ok=True)

epochs = 30
data_dir = "../dataset/traindata/"
model_name = "resnet18"  # resnet18 resnet34 resnet50 , cbam
num_classes = 2
input_width = 224
input_height = 224
input_channels = 1

train_batchsize = 64
valid_batchsize = 32
pretrain = True
num_worker = 0
sqeeze_feature = True
regression = False
thresh_score = 0.5

def plot_acc(train_acc,valid_acc):

    plt.clf()
    plt.plot(train_acc,label="train acc")
    plt.plot(valid_acc,label="valid acc")
    plt.legend()
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    save_path = os.path.join(save_dir,"acc.png")
    plt.savefig(save_path,dpi=300)
    plt.close()


def plot_loss(train_loss,valid_loss):

    plt.plot(train_loss,label="train loss")
    plt.plot(valid_loss,label="valid loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss value")
    save_path = os.path.join(save_dir,"loss.png")
    plt.savefig(save_path,dpi=300)

import warnings
warnings.filterwarnings("ignore")


def getData(dType="train",dataDir="",bSize=8,shuffle=True,imgH=224,imgW=224,channels=3):
    if channels==1:
        transform = T.Compose([T.CenterCrop((imgH, imgW)),
                               T.RandomHorizontalFlip(p=0.5),
                               T.RandomVerticalFlip(p=0.5),
                               T.Resize((imgH, imgW)),
                               T.Grayscale(num_output_channels=1),
                               T.ToTensor()])  # img = 0-1
    else:
        transform = T.Compose([T.CenterCrop((imgH, imgW)),
                               T.RandomHorizontalFlip(p=0.5),
                               T.RandomVerticalFlip(p=0.5),
                               T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                               T.Resize((imgH, imgW)),
                               T.ToTensor()]
                              )  # img = 0-1

    if dType=="train":
        trainDSet = ImageFolder(dataDir + "/train/", transform=transform)
        dLoad = DataLoader(trainDSet,
                           batch_size=bSize,
                           shuffle=shuffle,
                           num_workers=num_worker)
        return dLoad
    if dType in ["test","valid"]:
        valDSet = ImageFolder(dataDir + "/valid/", transform=transform)
        dLoad = DataLoader(valDSet,
                           batch_size=bSize,
                           shuffle=False,
                           num_workers=0)
        return dLoad


def buildNet(class_num=2,name="resnet18",input_channels=3):
    if name=="resnet18":
        model = models.resnet18(pretrained=pretrain)
        if input_channels!=3:
            model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, class_num)
        )
    if name=="resnet34":
        model = models.resnet34(pretrained=pretrain)
        if input_channels!=3:
            model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, class_num)
        )
    if name=="resnet50":
        model = models.resnet50(pretrained=pretrain)
        if input_channels!=3:
            model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, class_num)
        )
    print(model)
    return model


def train():



    # train_data, valid_data = data_loader()
    train_data = getData("train", dataDir=data_dir, bSize=train_batchsize, imgH=input_height, imgW=input_width,
                        channels=input_channels)
    valid_data = getData("test", dataDir=data_dir, bSize=valid_batchsize, imgH=input_height, imgW=input_width,
                       channels=input_channels)


    model = buildNet(class_num=num_classes,name=model_name,input_channels=input_channels)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device=",device)


    model = model.to(device)


    # if regression:
    #     loss_fc = torch.nn.MSELoss()
    # else:
    loss_fc = torch.nn.CrossEntropyLoss()

    # https://blog.csdn.net/tcn760/article/details/123965374
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    if sqeeze_feature:
        optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 0
    for epoch in range(epochs):

        model.train()
        train_correct = 0
        train_samle = 0
        train_epoch_loss = 0
        train_bn = 0

        for sample_batch in tqdm(train_data):
            inputs = sample_batch[0]
            train_labels = sample_batch[1]
            # GPU/CPU
            inputs = inputs.to(device)
            train_labels = train_labels.to(device)
            # if regression:
            #     inputs = inputs.to(torch.float32)
            #     train_labels = train_labels.to(torch.float32)


            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(train_labels)

            optimizer.zero_grad()

            train_outputs = model(inputs) #  out shape：[1,num_classes]
            loss = loss_fc(train_outputs, train_labels) # 计算损失

            # if regression:
            #     train_prediction = train_outputs.cpu().detach().numpy()
            #     train_prediction[np.where(train_prediction > thresh_score)] = 1
            #     train_prediction[np.where(train_prediction <= thresh_score)] = 0
            #     train_prediction = np.squeeze(train_prediction)
            #     train_prediction = torch.Tensor(train_prediction).to(device)
            # else:
            _, train_prediction = torch.max(train_outputs, 1)
            train_correct += (torch.sum((train_prediction == train_labels))).item()
            train_samle += train_labels.size(0)

            train_epoch_loss += loss.cpu().detach().numpy()
            train_bn += 1


            loss.backward()
            optimizer.step()

        train_acc.append(train_correct / train_samle)
        train_loss.append(train_epoch_loss / train_bn)


        model.eval()
        valid_correct = 0
        valid_samle = 0
        valid_epoch_loss = 0
        valid_bn = 0
        with torch.no_grad():
            for sample_batch in tqdm(valid_data):

                images_test, labels_test = sample_batch
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = model(images_test)

                loss2 = loss_fc(outputs_test, labels_test.long())

                # if regression:
                #     prediction = outputs_test.cpu().detach().numpy()
                #     prediction[np.where(prediction > thresh_score)] = 1
                #     prediction[np.where(prediction <= thresh_score)] = 0
                #     prediction = np.squeeze(prediction)
                #     prediction = torch.Tensor(prediction).to(device)
                # else:
                _, prediction = torch.max(outputs_test, 1)
                valid_correct += (torch.sum((prediction == labels_test))).item()
                valid_samle += labels_test.size(0)
                valid_epoch_loss += loss2.item()
                valid_bn += 1


        valid_acc.append(valid_correct / valid_samle)
        valid_loss.append(valid_epoch_loss / valid_bn)

        print("【{}/{}】【训练损失{:.4f}】【训练准确率{:.4f}】【验证损失{:.4f}】【验证准确率{:.4f}】".format(
            epoch+1,epochs,
            train_loss[-1],train_acc[-1],valid_loss[-1],valid_acc[-1]
        ))

        if valid_acc[-1]>best_acc:
            best_acc = valid_acc[-1]
            weigths_path = os.path.join(save_dir, "best.pth")
            torch.save(model, weigths_path)


    weigths_path = os.path.join(save_dir,"last.pth")
    torch.save(model, weigths_path)
    torch.cuda.empty_cache()


    plot_acc(train_acc,valid_acc)
    plot_loss(train_loss,valid_loss)

    print('training finish !')



if __name__ == '__main__':
    train()
    # predict()