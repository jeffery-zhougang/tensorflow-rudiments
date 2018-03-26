# _*_ coding=UTF-8 -*-

'''
Created on Jan 23, 2018

@author: Administrator
'''

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


# 导入MNIST数据集
mnist = input_data.read_data_sets('../dataset/MNIST', one_hot=True)


show_num = 8
for i in range(8):
    plt.subplot(str(441+i))
    img = mnist.train.images[i]
    label = mnist.train.labels[i]
    img.shape = [28,28]
    label.shape = [-1, 10]
    #print(img.shape)
    #print(label.shape)
    plt.title(np.argmax(label))
    plt.imshow(img,cmap='gray')
plt.show()    