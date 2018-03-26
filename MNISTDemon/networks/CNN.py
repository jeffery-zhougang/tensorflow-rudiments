# _*_ coding=UTF-8 -*-

'''
Created on 2017-09-05

@author: ex-zhougang001
'''

import os


import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.ops import control_flow_ops
from tensorflow.examples.tutorials.mnist import input_data


# 导入MNIST数据集
mnist = input_data.read_data_sets('../dataset/MNIST', one_hot=True)


input_size = 28

class_num = 10
# 权重衰减速率
weight_decay=0.0001
# BN的衰减速率
batch_norm_decay=0.997
# BN的epsilon默认1e-5
batch_norm_epsilon=1e-5
batch_norm_scale=True
# 学习率
learning_rate=0.001
# 学习率衰减
learning_rate_decay=0.97
# 学习率衰减步数
learning_rate_decay_steps=2000

def build_inputs():
    
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

    images = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='image_batch')

    labels = tf.placeholder(dtype=tf.int64, shape=[None, class_num], name='label_batch')
    
    return images, labels, keep_prob


def build_graph(x, y, keep_prob, is_training=True):
    
    # with tf.device('/cpu:0'):
    
    images = tf.reshape(x, [-1, input_size, input_size, 1])
    
    # 定义batch normalization（标准化）的参数字典
    batch_norm_params = { 
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    
    # 定义VGG-16网络结构
    # scale-1
    # [3*3*64 conv]
    # max pool
    # output 16*16*32
    conv1_1 = slim.conv2d(images, 32, [3, 3], 1, normalizer_fn=slim.batch_norm, 
                         normalizer_params=batch_norm_params, padding='SAME', scope='conv1_1')
    conv1_2 = slim.conv2d(conv1_1, 32, [3, 3], 1, normalizer_fn=slim.batch_norm, 
                         normalizer_params=batch_norm_params, padding='SAME', scope='conv1_2')
    
    max_pool_1 = slim.max_pool2d(conv1_2, [2, 2], [2, 2], padding='SAME')
    #dropout_1 = tf.nn.dropout(tf.nn.relu(max_pool_1), keep_prob=keep_prob)
    
    # scale-2
    # [3*3*128 conv]
    # max pool
    # output 8*8*64
    conv2_1 = slim.conv2d(max_pool_1, 64, [3, 3], 1, normalizer_fn=slim.batch_norm, 
                         normalizer_params=batch_norm_params, padding='SAME', scope='conv2_1')
    conv2_2 = slim.conv2d(conv2_1, 64, [3, 3], 1, normalizer_fn=slim.batch_norm, 
                         normalizer_params=batch_norm_params, padding='SAME', scope='conv2_2')
    max_pool_2 = slim.max_pool2d(conv2_2, [2, 2], [2, 2], padding='SAME')
    #dropout_2 = tf.nn.dropout(tf.nn.relu(max_pool_2), keep_prob=keep_prob)

    
    # scale-4
    # full connect
    flatten = slim.flatten(max_pool_2)
    fc = slim.fully_connected(flatten, 256, normalizer_fn=slim.batch_norm, 
                         normalizer_params=batch_norm_params, activation_fn=tf.nn.relu, scope='fc')
    logits = slim.fully_connected(slim.dropout(fc, keep_prob=keep_prob), class_num, activation_fn=None, scope='logits')
    # logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
    #print(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y, 1)))
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))
    
    global_step = tf.Variable(0, trainable=False)

    # learning rate decay
    rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=learning_rate_decay_steps, 
                                      decay_rate=learning_rate_decay, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=rate);
    # slim使用slim.batch_norm需要增加的条件
    train_op = slim.learning.create_train_op(loss, opt, global_step=global_step)
    
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)
    
    # predict label
    probabilities = tf.nn.softmax(logits)

    # save logs
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image("image", images)
    merged_summary_op = tf.summary.merge_all()
    
    # predict and accuracy in top_k
    #predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    #accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    
    return {'images': images,
            'labels': y,
            'keep_prob': keep_prob,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'merged_summary': merged_summary_op,
            'probabilities': probabilities
            }

    
# 定义训练参数
# 批量大小
_batch_size=128
# 验证步长
val_step = 1
# 模型保存不步长
save_step = 1000
# 运行多少个bacth
num_step = 2000
# 模型参数保存目录
checkpoint_dir = '../checkpoint/CNN'


def train():
    
    with tf.Session() as sess:
        
        print('============start trainning============')
        
        inputs, label, keep_prob = build_inputs()
        
        graph = build_graph(inputs, label, keep_prob)
        
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        
        train_writer = tf.summary.FileWriter('../train/CNN')
        
        step = 0
        
        for i in range(num_step):
            print(i)
            print(num_step)
            images, labels = mnist.train.next_batch(_batch_size)
            print(1)
            #print(labels)
            #print(images.shape)
            # 使用测试集验证准确率
            if (i+1)%val_step == 0:
                accuracy, loss = sess.run([graph["accuracy"], graph["loss"]], feed_dict={
                    inputs:mnist.test.images, label: mnist.test.labels, keep_prob: 1.0})
                # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
                print("epochs: %d, step: %d, val accuracy: %g, val loss: %g" % (mnist.train.epochs_completed, (i+1), accuracy, loss))
                print('end1')
            # 保存模型参数
            print(2)
            if (i+1)%save_step == 0:
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                saver.save(sess, os.path.join(checkpoint_dir,'cnntest'), global_step=step)
                print("============save model============")
            print(3)
            step, summary_str, _ = sess.run([graph["global_step"],
                                             graph["merged_summary"],
                                             graph["train_op"]],
                                            feed_dict={inputs: images, 
                                                       label: labels, 
                                                       keep_prob: 0.5})
            # 生成日志
            train_writer.add_summary(summary_str, step)
            

def inference(images, showimg=True):
    
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

        probabilities = sess.run([graph['probabilities']],
                                          feed_dict={inputs: images, keep_prob: 1.0})
        
        print('probabilities: ', probabilities)
            


def showconvimage(conv, shape=(4, 8)):
    
    rows = shape[0]
    cols = shape[1]
    _, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            img = conv[:,:,idx]
            axs[i, j].imshow(img, cmap='gray')
            idx = idx+1
    plt.show()


if __name__ == '__main__':
    
    op = 'train'
    #op = 'inference'
    
    if op == 'train':
        
        train()
        print('end')
    else:
        
        img = mnist.train.images[2]
        
        label = mnist.train.labels[2]
        
        img.shape = [-1, 784]
        label.shape = [-1, 10]
        
        plt.imshow(img.reshape([28,28]),cmap='gray')
        plt.show()
        
        inference(img)













