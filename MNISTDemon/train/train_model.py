# _*_ coding=UTF-8 -*-

'''
Created on Mar 26, 2018

@author: jeffery.zhougang
'''

import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from networks.cnn import CNN as network

# 加载MNIST数据集
mnist = input_data.read_data_sets('..\\dataset\\MNIST', one_hot=True)
print('loading data done......')


# 训练参数
lr=0.001
class_num=10
batch_size=64
num_step=100
val_step=10

save_dir='..\\checkpoint\\cnn'
log_dir='..\\log\\cnn'
with tf.Session() as sess:

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='image_batch')

    labels = tf.placeholder(dtype=tf.int64, shape=[None, class_num], name='label_batch')

    model = network(inputs)
    
    logits = model.get_output('logits')
    
    loss = model.loss(logits, labels)
    accuracy = model.accuracy(logits, labels)
    train_op = model.train_op(lr, loss)
    summary = model.summary(loss, accuracy)
    # 初始化参数
    sess.run(tf.global_variables_initializer())
    
    # 初始化SAVER对象和日志对象
    saver = tf.train.Saver(tf.global_variables())
    train_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    
    step=1
    print('===== start training =====')
    while step<=num_step:
    #for step in range(num_step):
        
        # 验证模型
        if step%val_step == 0:
            acc, err = sess.run([accuracy, loss], feed_dict={inputs: mnist.test.images, labels: mnist.test.labels})
            print("epoch: %d, step: %d, accuracy: %g, loss: %g" % (mnist.train.epochs_completed, step, acc, err))
        
        # 训练一个batch
        images, true_labels = mnist.train.next_batch(batch_size)
        _, summary_str = sess.run([train_op, summary], feed_dict={inputs: images, labels: true_labels})
        
        # 记录日志
        train_writer.add_summary(summary_str, step)
        
        step+=1
        
    # 最后保存模型    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    saver.save(sess, os.path.join(save_dir,'mnist-cnn'))




