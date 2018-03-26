# _*_ coding=UTF-8 -*-

'''
Created on Mar 26, 2018

@author: jeffery.zhougang
'''

import tensorflow as tf

import tensorflow.contrib.slim as slim

from networks.network_mnist import NetWork

class CNN(NetWork):
    
    def setup(self):
        
        inputs = tf.reshape(self.inputs, shape=(-1, 28, 28, 1))
        x=slim.conv2d(inputs, 32, [3, 3], [1, 1], padding='SAME', scope='conv1_1')
        self.set_layer(x, 'conv1_1')
        x=slim.conv2d(x, 32, [3, 3],  [1, 1], padding='SAME', scope='conv1_2')
        self.set_layer(x, 'conv1_2')
        x=slim.max_pool2d(x, [2, 2], [2, 2], padding='VALID', scope='pool1')
        self.set_layer(x, 'pool1')
        
        x=slim.conv2d(x, 64, [3, 3], [1, 1], padding='SAME', scope='conv2_1')
        self.set_layer(x, 'conv2_1')
        x=slim.conv2d(x, 64, [3, 3],  [1, 1], padding='SAME', scope='conv2_2')
        self.set_layer(x, 'conv2_2')
        x=slim.max_pool2d(x, [2, 2], [2, 2], padding='VALID', scope='pool2')
        self.set_layer(x, 'pool2')
        
        x=slim.flatten(x, scope='flatten')
        self.set_layer(x, 'flatten')
        
        x=slim.fully_connected(x, 256, scope='fc')
        self.set_layer(x, 'fc')
        
        logits=slim.fully_connected(x, 10, activation_fn=None, scope='logits')
        self.set_layer(logits, 'logits')
        