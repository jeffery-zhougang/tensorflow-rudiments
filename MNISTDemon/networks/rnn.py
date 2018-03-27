# _*_ coding=UTF-8 -*-

'''
Created on Jan 17, 2018

@author: jeffery.zhougang
'''
import tensorflow as tf

from tensorflow.contrib import rnn

from networks.network_mnist import NetWork

class RNN(NetWork):
    
    def setup(self):
        
        inputs = tf.reshape(self.inputs, shape=(-1, 28, 28))
        
        def getLSTMCell():
            lstm_cell = rnn.BasicLSTMCell(num_units=256, forget_bias=1.0, state_is_tuple=True)
            return lstm_cell

    
        mlstm_cell = rnn.MultiRNNCell([getLSTMCell() for _ in range(2)], state_is_tuple=True)
        self.set_layer(mlstm_cell, 'rnncell')
        
        outputs, _ = tf.nn.dynamic_rnn(mlstm_cell, inputs=inputs, time_major=False, dtype=tf.float32)
        
        # 只取最后一个序列的输出结果
        h_state = outputs[:, -1, :]
        self.set_layer(h_state, 'laststate')
        
        W = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1), dtype=tf.float32, trainable=True)
        bias = tf.Variable(tf.constant(0.1,shape=[10]), dtype=tf.float32, trainable=True)
        
        logits = tf.matmul(h_state, W) + bias
        self.set_layer(logits, 'logits')
        
        