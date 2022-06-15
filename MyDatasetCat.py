#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 11:30
# @Author  : lanlin
# reference: https://blog.csdn.net/qq_40788447/article/details/114937779
#            https://blog.csdn.net/full_speed_turbo/article/details/102700226

import h5py
import torch
import numpy as np
import torchvision.transforms as transforms


# 若应用transforms.ToTensor()，则可不用np.transpose((0, 3, 2, 1)) 和 torch.from_numpy
# x_train = train_set_x_orig.transpose((0, 3, 2, 1))  # 209*64*64*3——209*3*64*64
# y_train = train_set_y_orig  # 模拟对应样本的标签， 209*1个标签
# x_test = test_set_x_orig.transpose((0, 3, 2, 1))  # 50*64*64*3——>50*3*64*64
# y_test = test_set_y_orig  # 模拟对应样本的标签， 50*1个标签
# x_train = torch.from_numpy(x_train)
# y_train = torch.from_numpy(y_train)
# x_test = torch.from_numpy(x_test)
# y_test = torch.from_numpy(y_test)


class Mydataset(torch.utils.data.Dataset):

    def __init__(self, x, y, n_feature=10, transform=None):
        self.x = x
        self.y = y
        self.idx = list()
        self.transform = transform
        self.n_feature = n_feature
        for item in x:  # 提取x的10个数据[item,:]
            self.idx.append(item)  # 注意要用list封装（数据或者数据路径）
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        if self.transform:
            input_data = self.transform(input_data)
        return input_data, target  # 返回单个样本对（数据、标签）

    def __len__(self):
        return len(self.idx)


if __name__ ==('__main__'):

    # 设置超参数
    image_size = 64
    image_channel = 3
    lr = 0.02  # 学习率
    epochs = 100  # 训练轮数
    n_feature = 64 * 64 * 3  # 输入特征
    n_hidden = 2000  # 隐层节点数
    n_output = 2  # 输出(是否为猫2种类别)

    # Loading the data (cat/non-cat)
    train_dataset = h5py.File('./dataset_h5/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # unit8类型  train set data
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # int64类型  train set labels

    test_dataset = h5py.File('./dataset_h5/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # unit8类型  test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # int64类型  test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    datasets = Mydataset(train_set_x_orig, train_set_y_orig,
                         n_feature, transform=transform)  # 初始化
    mydataloader = torch.utils.data.DataLoader(datasets, batch_size=4, num_workers=0)

    for i, (input_data, target) in enumerate(mydataloader):
        print('input_data%d' % i, input_data)
        print('target%d' % i, target)


