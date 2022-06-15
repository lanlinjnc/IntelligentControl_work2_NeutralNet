#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 23:32
# @Author  : lanlin


import cv2
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 超参数
image_size = 64
image_channel = 3
learning_rate = 0.001
num_epochs = 100  # 训练轮数
n_feature = 64 * 64 * 3  # 输入特征
m_hidden = 209  # 隐层节点数
s_output = 2  # 输出(是否为猫2种类别)

# 加载数据 (cat/non-cat)
train_dataset = h5py.File('./dataset_h5/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]).astype(np.float32)
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], 64*64*3)
train_set_x = torch.from_numpy(train_set_x)  # 转成tensor
train_set_y = np.array(train_dataset["train_set_y"][:]).astype(np.int64)

test_dataset = h5py.File('./dataset_h5/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]).astype(np.float32)
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], 64*64*3)
test_set_x = torch.from_numpy(test_set_x)  # 转成tensor
test_set_y = np.array(test_dataset["test_set_y"][:]).astype(np.int64)

classes = np.array(test_dataset["list_classes"][:])  # the list of classes

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mydataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        super(Mydataset, self).__init__()
        self.x = x
        self.y = y
        self.idx = list()
        for item in self.x:  # 提取x的10个数据[item,:]
            self.idx.append(item)  # 注意要用list封装（数据或者数据路径）
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target  # 返回单个样本对（数据、标签）

    def __len__(self):
        return len(self.idx)


class RBFN(nn.Module):
    """
    以高斯核作为径向基函数
    """

    def __init__(self, n_feature, m_hidden, s_output=2):
        """
        :param centers: shape=[center_num,data_dim]
        :param s_output:
        """
        super(RBFN, self).__init__()
        self.n_feature = n_feature  # 64*64*3
        self.m_hidden = m_hidden  # 2000
        self.s_output = s_output  # 2
        self.center = train_set_x.clone().to(device)  # 深拷贝
        self.sigma = 1000*torch.ones(m_hidden, dtype=torch.float32).to(device)  # 不参加训练

        self.linear = nn.Linear(self.m_hidden, self.s_output, bias=True)
        self.initialize_weights()  # 创建对象时自动执行

    def forward(self, input_data):
        ''' 要求input_one_data维度为[batch, n_feature] '''
        radial_val = self.RBF_caculate(input_data, self.center)
        class_score = self.linear(radial_val)
        return class_score

    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 50)
                m.bias.data.zero_()
                # m.bias.data.normal_(0, 50)

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

    def RBF_caculate(self, X, C):
        result = torch.zeros((X.shape[0], m_hidden)).to(device)  # [batch, 209]

        for i in range(X.shape[0]):  # 径向基函数计算
            temp_result = X[i] - C  # [209,12288]
            temp_result = torch.norm(temp_result, dim=1)  # [209,12288]
            temp_result = temp_result / torch.pow(self.sigma, 2)
            temp_result = torch.exp(-temp_result)  # [209,]
            result[i,:] = temp_result

        return result


# 更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def tensor_float_np_unit8(image):

    image_max = np.max(image)
    image_min = np.min(image)
    image_save = 255*(image-image_min)/(image_max-image_min)
    image_save = image_save.reshape((64, 64, 3)).astype(np.uint8)

    return image_save


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


def detect_local_image(model, image_path, model_paras_path, n_feature):
    # read image
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
    x_image = image.reshape((1, n_feature))
    x_image = torch.from_numpy(x_image).to(device)

    # read model
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)

    pre = model(x_image)
    # print(pre)
    if pre[0][0] > pre[0][1]:
        print("is cat")
    else:
        print("not cat")



if __name__ == "__main__":

    # 数据载入
    datasets_train = Mydataset(train_set_x, train_set_y)
    datasets_test = Mydataset(test_set_x, test_set_y)
    train_loader = torch.utils.data.DataLoader(datasets_train, 1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(datasets_test, 1, shuffle=False)

    model = RBFN(n_feature, m_hidden, s_output).to(device)
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

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_train_step, loss.item()))

            with torch.no_grad():  # 下面是没有梯度的计算,主要是训练集统计使用，不需要再计算梯度了
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
    torch.save(model.state_dict(), './weights/RBF_params.pth')

    # 绘制损失函数和精度
    fig_name = "cat_dataset_classify_RBF"
    draw_result(fig_name, accuracy_train_epoch, accuracy_test_epoch, loss_train_epoch)

    # 逐图片检验最终模型上的训练集训练效果
    model.eval()
    print('---------最终模型上的训练集训练效果-----------')
    with torch.no_grad():  # 下面是没有梯度的计算,主要是训练集统计使用，不需要再计算梯度了
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

    # 本地磁盘单张图片检测
    model = RBFN(n_feature, m_hidden, s_output).to(device)  # 调用网络
    model.eval()
    image_path = './internet_image/cat_internet1.jpg'
    model_paras_path = './weights/RBF_params.pth'
    detect_local_image(model, image_path, model_paras_path, n_feature)
