# _*_ coding=UTF-8 -*-

'''
Created on Jan 24, 2018

@author: jeffery.zhougang
'''

import tensorflow as tf

from networks.network_mnist import NetWork

class LR(NetWork):
    
    def setup(self):
        # Weights & bias
        weights1 = tf.Variable(tf.random_normal([784, 512]))
        bias1 = tf.Variable(tf.random_normal([512]))
        layer1 = tf.add(tf.matmul(self.inputs, weights1), bias1)
        self.set_layer(layer1, 'layer1')
        
        weights2 = tf.Variable(tf.random_normal([512, 128]))
        bias2 = tf.Variable(tf.random_normal([128]))
        layer2 = tf.add(tf.matmul(layer1, weights2), bias2)
        self.set_layer(layer2, 'layer2')
        
        weights3 = tf.Variable(tf.random_normal([128, 10]))
        bias3 = tf.Variable(tf.random_normal([10]))
        
        logits = tf.add(tf.matmul(layer2, weights3), bias3)
        self.set_layer(logits, 'logits')
        
        