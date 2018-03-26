# _*_ coding=UTF-8 -*-

'''
Created on Mar 26, 2018

@author: jeffery.zhougang
'''

import tensorflow as tf


class NetWork(object):
    
    def __init__(self, inputs, isTrain=True):
        print('NetWork init......')
        self.inputs = inputs
        self.isTrain = isTrain
        self.layers = dict()
        self.setup()
        
    def setup(self):
        raise NotImplementedError('Must be implemented by subclass.')
    
    def set_layer(self, layer, name):
        self.layers[name] = layer
    
    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            raise KeyError('Unknown layer name: %s'%layer)
        
        return layer
    
    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(labels, 1)), name='loss')
        return loss
    
    def accuracy(self, logits, labels):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32), name='accuracy')
        return accuracy
    
    def train_op(self, lr, loss, optimizer=tf.train.AdamOptimizer):
        train_op = optimizer(lr).minimize(loss)
        return train_op
    
    def summary(self, *args):
        
        for tensor in args:
            tf.summary.scalar(tensor.name, tensor)
        return tf.summary.merge_all()
    