#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 21:20
# @Author  : lanlin


import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn.functional as Fun


def image_standardize(image):
    ''' 要求输入的图像尺寸为[batch,w,h,c]或 [w,h,c]'''
    result = np.zeros(image.shape())
    flag = len(image.shape())

    pass


# 定义BP神经网络
class BPNetModel(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPNetModel, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐层网络
        self.dropout = torch.nn.Dropout(p=0.5)  # p=0.3表示神经元有p = 0.3的概率不被激活
        self.BN1 = torch.nn.BatchNorm1d(num_features=n_feature)
        self.BN2 = torch.nn.BatchNorm1d(num_features=n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络

    def forward(self, x):
        x = self.BN1(x)
        x = Fun.relu(self.hiddden(x))  # 隐层激活函数采用relu()函数
        x = self.BN2(x)
        x = self.dropout(x)
        out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        return out


def detect_image(model, image_path, model_paras_path, n_feature):
    # read image
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('temp_win', image)
    x_image = image.reshape((1, n_feature))

    x_image = torch.from_numpy(x_image)
    # x_image = torch.unsqueeze(x_image, 0)

    # read model
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)
    model.eval()
    pre = model(x_image)
    print(pre)
    if pre[0][0] > pre[0][1]:
        print("is cat")
    else:
        print("not cat")


if __name__ == "__main__":

    # 设置超参数
    image_size = 64
    image_channel = 3
    lr = 0.01  # 学习率
    epochs = 100  # 训练轮数
    n_feature = 64 * 64 * 3  # 输入特征
    n_hidden = 2000  # 隐层节点数
    n_output = 2  # 输出(是否为猫2种类别)

    # 准备数据，Loading the data (cat/non-cat)
    train_dataset = h5py.File('./dataset_h5/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]).astype(np.float32)  # train set features
    train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], n_feature))  # 将图片拉直成一维向量
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]).astype(np.int64)  # train set labels

    test_dataset = h5py.File('./dataset_h5/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]).astype(np.float32)  # test set features
    test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], n_feature))  # 将图片拉直成一维向量
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]).astype(np.int64)  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # 将数据类型转换为tensor
    x_train = torch.tensor(train_set_x_orig)
    y_train = torch.tensor(train_set_y_orig)
    x_test = torch.tensor(test_set_x_orig)
    y_test = torch.tensor(test_set_y_orig)

    # 定义优化器和损失函数
    net = BPNetModel(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)  # 初始化网络
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用Adam优化器，并设置学习率
    loss_fun = torch.nn.CrossEntropyLoss()  # 注意使用交叉熵损失函数时标签只需为1D数字(0,1,2...)即可

    # 训练数据
    loss_steps = np.zeros(epochs)  # 构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
    accuracy_steps = np.zeros(epochs)

    for epoch in range(epochs):
        net.train()
        y_pred = net(x_train)  # 前向传播
        loss = loss_fun(y_pred, y_train)  # 预测值和真实值对比
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度
        loss_steps[epoch] = loss.item()  # 保存loss
        running_loss = loss.item()
        print(f"第{epoch}次训练，loss={running_loss}".format(epoch, running_loss))
        with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
            net.eval()
            y_pred = net(x_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_steps[epoch] = correct.mean()
            print("测试是否为猫的预测准确率", accuracy_steps[epoch])

    torch.save(net.state_dict(), './weights/BP_params.pth')

    # 绘制损失函数和精度
    fig_name = "cat_dataset_classify_BPNet"
    fontsize = 15
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 12), sharex=True)
    ax1.plot(accuracy_steps)
    ax1.set_ylabel("test accuracy", fontsize=fontsize)
    ax1.set_title(fig_name, fontsize="xx-large")
    ax2.plot(loss_steps)
    ax2.set_ylabel("train lss", fontsize=fontsize)
    ax2.set_xlabel("epochs", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./results/' + fig_name + '.png')
    plt.show()

    # 图片检测
    model = BPNetModel(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)  # 调用网络
    model.eval()
    image_path = './internet_image/rand_internet1.jpg'
    model_paras_path = './weights/BP_params.pth'
    detect_image(model, image_path, model_paras_path, n_feature)
