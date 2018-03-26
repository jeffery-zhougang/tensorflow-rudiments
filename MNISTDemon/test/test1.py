# _*_ coding=UTF-8 -*-

'''
Created on Jan 25, 2018

@author: Administrator
'''
# Solution is available in the other "solution.py" tab
import tensorflow as tf

slim = tf.contrib.slim



def atrousConv():
    #img = tf.ones((1, 3,3,3), dtype=tf.float32)
    '''img = tf.constant(value=[
        [[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]
    ],dtype=tf.float32)'''
    
    img = tf.constant(value=1, shape=[1,3,3,3], dtype=tf.float32)
    filter = tf.constant(value=1, shape=[3,3,3,1], dtype=tf.float32)
    out = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='SAME')
    
    #out = slim.conv2d(img, 1, [3,3])
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        # output = sess.run(softmax,    )
        output = sess.run(out)
    return output
    

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    # softmax =     
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        # output = sess.run(softmax,    )
        output = sess.run(tf.nn.softmax(logits), {logits: logit_data})
    return output
output = atrousConv()
print(output.shape)
print(output)