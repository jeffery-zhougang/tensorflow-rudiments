# _*_ coding=UTF-8 -*-

'''
Created on Jan 17, 2018

@author: Administrator
'''


import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data



# 导入MNIST数据集
mnist = input_data.read_data_sets('../dataset/MNIST', one_hot=True)


# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 28
# 每个隐含层的节点数
hidden_size = 256
# LSTM layer 的层数
layer_num = 2
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 10
# 学习率
lr = 1e-3


# 构建输入
def build_inputs():
    
    inputs = tf.placeholder(tf.float32, [None, 784])
    #batch_size = tf.placeholder(tf.int32, [])
    label = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    
    return inputs, label, keep_prob


# 构建模型
def build_graph(inputs, label, keep_prob):
    
    inputs = tf.reshape(inputs, [-1, None, 28])
    
    global_step = tf.Variable(0, trainable=False)
    
    def getLSTMCell():
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        return lstm_cell

    
    mlstm_cell = rnn.MultiRNNCell([getLSTMCell() for _ in range(layer_num)], state_is_tuple=True)

    # 用全零来初始化state
    #init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=inputs, time_major=False, dtype=tf.float32)
    
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
    
    '''print('outputs: ', outputs)
    print('state: ', state)
    print('h_state: ', h_state)'''
    
    '''outputs = list()
    state = init_state
    print(state)
    print(X.shape)
    with tf.variable_scope('RNN'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态
            (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
            outputs.append(cell_output)
    h_state = outputs[-1]
    
    print(h_state)'''
    
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32, trainable=True)
    #print('W: ', W)
    bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32, trainable=True)
    #print('bias: ', bias)
    probabilities = tf.nn.softmax(tf.matmul(h_state, W) + bias)
    #print('probabilities: ', probabilities)
    
    #print(label)
    # 损失和评估函数
    loss = -tf.reduce_mean(label * tf.log(probabilities))
    #print('loss: ', loss)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    
    correct_prediction = tf.equal(tf.argmax(probabilities,1), tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # 记录日志
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summay = tf.summary.merge_all()
    
    return {"global_step": global_step,
            "outputs": outputs,
            "state": state,
            "h_state": h_state,
            "weight": W,
            "bias": bias,
            "probabilities": probabilities,
            "loss": loss,
            "train_op": train_op,
            "accuracy": accuracy,
            "merged_summay": merged_summay}   
    

# 定义训练参数
# 批量大小
_batch_size = 128
# 验证步长
val_step = 200
# 模型保存不步长
save_step = 1000
# 运行多少个bacth
num_step = 2000
# 模型参数保存目录
checkpoint_dir = '../checkpoint/lstm'


def train():
    
    with tf.Session() as sess:
        
        print('============start trainning============')
        
        inputs, label, keep_prob = build_inputs()
        
        graph = build_graph(inputs, label, keep_prob)
        
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        
        train_writer = tf.summary.FileWriter('../train/lstm')
        
        step = 0
        
        for i in range(num_step):
            
            batch = mnist.train.next_batch(_batch_size)
            
            # 使用测试集验证准确率
            if (i+1)%val_step == 0:
                accuracy = sess.run(graph["accuracy"], feed_dict={
                    inputs:mnist.test.images, label: mnist.test.labels, keep_prob: 1.0})
                # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
                print("epochs: %d, step: %d, val accuracy: %g" % (mnist.train.epochs_completed, (i+1), accuracy))
            
            # 保存模型参数
            if (i+1)%save_step == 0:
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                saver.save(sess, os.path.join(checkpoint_dir,'lstmtest'), global_step=step)
                print("============save model============")
            
            step, summary_str, _ = sess.run([graph["global_step"],
                                             graph["merged_summay"],
                                             graph["train_op"]],
                                            feed_dict={inputs: batch[0], 
                                                       label: batch[1], 
                                                       keep_prob: 0.5})
            # 生成日志
            train_writer.add_summary(summary_str, step)

def inference(x, y):
    
    with tf.Session() as sess:

        print('============start inference============')

        inputs, label, keep_prob= build_inputs()
        
        graph = build_graph(inputs, label, keep_prob)

        saver = tf.train.Saver()

        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        
        # 从最后的保存掉加载已训练参数
        if ckpt:
            print('============restore session from {}============'.format(ckpt))
            saver.restore(sess, ckpt)
            
        probabilities = sess.run(graph["probabilities"], feed_dict={
                    inputs:x, label: y, keep_prob: 1.0})
        
        print('probabilities: ', probabilities)
        print('numer:', np.argmax(probabilities))
'''
def showStep():
    import matplotlib.pyplot as plt
    
    print(mnist.train.labels[4])
    
    X3 = mnist.train.images[4]
    img3 = X3.reshape([28, 28])
    plt.imshow(img3, cmap='gray')
    plt.show()
    
    X3.shape = [-1, 784]
    y_batch = mnist.train.labels[4]
    y_batch.shape = [-1, class_num]
    
    X3_outputs = np.array(sess.run(outputs, feed_dict={
                _X: X3, y: y_batch, keep_prob: 1.0, batch_size: 1}))
    print(X3_outputs.shape)
    X3_outputs.shape = [28, hidden_size]
    print(X3_outputs.shape)
    
    
    h_W = sess.run(W, feed_dict={
                _X:X3, y: y_batch, keep_prob: 1.0, batch_size: 1})
    h_bias = sess.run(bias, feed_dict={
                _X:X3, y: y_batch, keep_prob: 1.0, batch_size: 1})
    h_bias.shape = [-1, 10]
    
    bar_index = range(class_num)
    for i in range(X3_outputs.shape[0]):
        plt.subplot(7, 4, i+1)
        X3_h_shate = X3_outputs[i, :].reshape([-1, hidden_size])
        pro = sess.run(tf.nn.softmax(tf.matmul(X3_h_shate, h_W) + h_bias))
        plt.bar(bar_index, pro[0], width=0.2 , align='center')
        plt.axis('off')
    plt.show()'''


if __name__ == '__main__':
    
    op = 'inference'
    
    if op == 'train':
        
        train()
    else:
        
        img = mnist.train.images[7]
        
        label = mnist.train.labels[7]
        
        img.shape = [-1, 784]
        label.shape = [-1, 10]
        print(img)
        plt.imshow(img.reshape([28,28]),cmap='gray')
        plt.show()
        
        inference(img, label)

    



