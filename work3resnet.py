#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/3 23:32
# @Author  : lanlin
# reference: https://blog.csdn.net/weixin_41823298/article/details/108711064


import cv2
import h5py
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from MyDatasetCat import Mydataset
import torchvision.transforms as transforms


def conv3x3(in_channels, out_channels, stride=1):
    ''' 3x3 卷积定义 '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Resnet 的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet定义
class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet18, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.layer4 = self.make_layer(block, 128, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)  # [1,3,64,64]
        out = self.bn(out)  # [1,16,64,64]
        out = self.relu(out)
        out = self.layer1(out)  # [1,16,64,64]
        out = self.layer2(out)  # [1,32,32,32]
        out = self.layer3(out)  # [1,64,16,16]
        out = self.layer4(out)  # [1,128,8,8]
        out = self.avg_pool(out)  # [1,128,1,1]
        out = out.view(out.size(0), -1)  # [1,128]
        out = self.fc(out)  # [1,2]
        return out


# 更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def detect_local_image(model, image_path, model_paras_path, transform):
    # read image
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('temp_win', image)
    image = transform(image)  # totensor and normalize
    image = image.to(device)
    image = image.expand(1, 3, 64, 64)

    # read model
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)

    pre = model(image)
    # print(pre)
    if pre[0][0] < pre[0][1]:
        print("is cat")
    else:
        print("not cat")


def tensor_float_np_unit8(image):

    image_max = np.max(image)
    image_min = np.min(image)
    image_save = 255*(image-image_min)/(image_max-image_min)
    image_save = image_save.transpose((1, 2, 0)).astype(np.uint8)  # (H x W x C)

    return image_save


def save_error_image(model, images, labels, model_paras_path, transform=None):

    # read model
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)

    # detect image
    for i in range(images.shape[0]):
        # read image
        image = images[i,:,:,:].copy()  # 注意要深拷贝
        label = labels[i].copy()
        # print(label)
        label = torch.tensor(label)  # 注意要转成tensor
        if transform:
            image = transform(image)  # (H x W x C)————>(C x H x W)
        image = image.to(device)
        label = label.to(device)
        image = torch.unsqueeze(image, 0)
        pre = model(image)
        _, pre_location = torch.max(pre.data, 1)
        # print(pre_location)
        if (pre[0][0]<pre[0][1]) and (label==1):
            print("pre is cat, and label is cat")
            image = torch.squeeze(image, 0)
            image_save = image.clone().detach().cpu().numpy()  # (C x H x W)

            image_save = tensor_float_np_unit8(image_save)  # (H x W x C)
            filename = './results/TP/test_image' + str(i) + '.jpg'
            # filename = './results/TP/train_image' + str(i) + '.jpg'
            cv2.imwrite(filename, image_save)
            pass
        elif (pre[0][0]<pre[0][1]) and (label==0):
            print("pre is cat, but label is not cat")
            image = torch.squeeze(image, 0)
            image_save = image.clone().detach().cpu().numpy()
            image_save = tensor_float_np_unit8(image_save)
            filename = './results/FP/test_image' + str(i) + '.jpg'
            # filename = './results/FP/train_image' + str(i) + '.jpg'
            cv2.imwrite(filename, image_save)
            pass
        elif (pre[0][0]>pre[0][1]) and (label==1):
            print("pre is not cat, but label is cat")
            image = torch.squeeze(image, 0)
            image_save = image.clone().detach().cpu().numpy()
            image_save = tensor_float_np_unit8(image_save)
            filename = './results/FN/test_image' + str(i) + '.jpg'
            # filename = './results/FN/train_image' + str(i) + '.jpg'
            cv2.imwrite(filename, image_save)
            pass
        else:
            print("pre is not cat, and label is not cat")
            image = torch.squeeze(image, 0)
            image_save = image.clone().detach().cpu().numpy()
            image_save = tensor_float_np_unit8(image_save)
            filename = './results/TN/test_image' + str(i) + '.jpg'
            # filename = './results/TN/train_image' + str(i) + '.jpg'
            cv2.imwrite(filename, image_save)
            pass


def draw_result(fig_name, accuracy_train_epoch, accuracy_test_epoch, loss_train_epoch):
    fontsize = 15
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 12), sharex=True)
    ax1.plot(accuracy_train_epoch)
    ax1.set_ylabel("train accuracy", fontsize=fontsize)
    ax1.set_title(fig_name, fontsize="xx-large")
    ax2.plot(loss_train_epoch)
    ax2.set_ylabel("train loss", fontsize=fontsize)
    ax3.plot(accuracy_test_epoch)
    ax3.set_ylabel("test accuracy", fontsize=fontsize)
    ax3.set_xlabel("epochs", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./results/' + fig_name + '.png')
    plt.show()


if __name__ == "__main__":

    # 超参数定义
    num_epochs = 80
    learning_rate = 0.01  # 初始学习率

    # 加载数据 (cat/non-cat)
    train_dataset = h5py.File('./dataset_h5/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # unit8类型  train set data
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # int64类型  train set labels

    test_dataset = h5py.File('./dataset_h5/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # unit8类型  test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # int64类型  test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # 判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理、增广函数定义等
    transform_train = transforms.Compose([
        transforms.ToTensor(),  # 一定要先转成tensor类型
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Pad(4),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
    ])

    # 测试图片时所用的的transform，不进行数据增广
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 一定要先转成tensor类型
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 数据载入
    datasets_train = Mydataset(train_set_x_orig, train_set_y_orig, transform=transform_train)
    datasets_test = Mydataset(test_set_x_orig, test_set_y_orig, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(datasets_train, batch_size=209, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=50, shuffle=False)

    # 可读取本地网络参数继续上次训练
    model = ResNet18(ResidualBlock, [2, 2, 2, 2]).to(device)  # 初始化调用网络
    # print(model)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 定义训练时保存的数据
    loss_train_epoch = np.zeros(num_epochs)  # 构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
    accuracy_train_epoch = np.zeros(num_epochs)
    accuracy_test_epoch = np.zeros(num_epochs)
    total_train_step = len(train_loader)
    curr_lr = learning_rate

    # 开始训练数据集
    for epoch in range(num_epochs):
        model.train()
        loss_train_steps = 0.0
        accuracy_train_steps = 0.0
        train_correct = 0
        train_total = 0
        for i, (images, labels) in enumerate(train_loader):  # 这一步才会运行数据预处理transform
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, pre_lacation = torch.max(outputs.data, 1)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train_steps += loss.item()  # 保存loss

            if (i + 1) % 1 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_train_step, loss.item()))

            with torch.no_grad():  # 下面是没有梯度的计算,主要是训练集统计使用，不需要再计算梯度了
                model.eval()  # 不可省略，因为有BN
                _, predicted = torch.max(outputs.data, 1)  # 这里保存最大值的索引
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        accuracy_train_steps = train_correct / train_total
        print('Accuracy of the epoch model on the train images: {} %'.format(100 * accuracy_train_steps))

        with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
            model.eval()
            label_test_correct = 0
            label_test_total = 0
            for i_test, (images_test, labels_test) in enumerate(test_loader):
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)
                outputs_test = model(images_test)
                _, predicted_test = torch.max(outputs_test.data, 1)  # 这里保存最大值的索引
                label_test_total += labels_test.size(0)
                label_test_correct += (predicted_test == labels_test).sum().item()
            accuracy_test_steps = label_test_correct / label_test_total
            print('Accuracy of the epoch model on the test images: {} %'.format(100 * accuracy_test_steps))

        accuracy_train_epoch[epoch] = accuracy_train_steps
        accuracy_test_epoch[epoch] = accuracy_test_steps
        loss_train_epoch[epoch] = loss_train_steps / total_train_step  # 保存loss

        # 延迟学习率
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    # S将模型保存
    torch.save(model.state_dict(), './weights/resnet_params.pth')

    # 绘制损失函数和精度
    fig_name = "cat_dataset_classify_ResNet18"
    draw_result(fig_name, accuracy_train_epoch, accuracy_test_epoch, loss_train_epoch)

    # 逐图片检验最终模型上的训练集训练效果
    print('---------最终模型上的训练集训练效果-----------')
    with torch.no_grad():  # 下面是没有梯度的计算,主要是训练集统计使用，不需要再计算梯度了
        model.eval()
        train_correct = 0
        train_total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 这里保存最大值的索引
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        accuracy_train_steps = train_correct / train_total
        print('Accuracy of the epoch model on the test images: {} %'.format(100 * accuracy_train_steps))
        print(predicted)

    # 保存分类错误的图片
    with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        model = ResNet18(ResidualBlock, [2, 2, 2, 2]).to(device)  # 调用网络
        model_paras_path = './weights/resnet_params.pth'
        save_error_image(model, test_set_x_orig, test_set_y_orig, model_paras_path, transform_test)

    # 本地磁盘单张图片检测
    with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        model = ResNet18(ResidualBlock, [2, 2, 2, 2]).to(device)  # 调用网络
        model.eval()
        image_path = './internet_image/cat_internet1.jpg'
        model_paras_path = './weights/resnet_params.pth'
        detect_local_image(model, image_path, model_paras_path, transform_test)

