#! /usr/bin/python

import numpy as np
import h5py
import torch
import cv2


# Loading the data (cat/non-cat)
train_dataset = h5py.File('./dataset_h5/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
# show_image(train_set_x_orig)
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels
# print(np.sum(train_set_y_orig))  # 72张真猫

test_dataset = h5py.File('./dataset_h5/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
# show_image(test_set_x_orig)
test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels
# print(np.sum(test_set_y_orig))  # 33张真猫

classes = np.array(test_dataset["list_classes"][:])  # the list of classes

train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


def show_data():
    print ("Dataset dimensions:")
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))


def show_image_label(image_array, label_array):
    for i in range(image_array.shape[0]):
        image_single = image_array[i,:,:]
        label_single = label_array[i]
        cv2.imshow(str(label_single),image_single)
        cv2.waitKey(500)


if __name__ == "__main__":
    # show_data()
    show_image_label(test_set_x_orig, test_set_y_orig)
