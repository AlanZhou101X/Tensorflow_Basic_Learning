# coding: utf-8

import tensorflow as tf
import numpy as np

init_op = tf.global_variables_initializer()

'''

# x = tf.placeholder("float", [10, 1])
# x = tf.constant([[1], [2]], dtype=tf.float32)

x_train = np.linspace(0, 2, 100, endpoint=True)   # 生成[0,2]区间100个点

x_t = np.zeros((len(x_train), 1))
for i in range(len(x_train)):
    x_t[i] = x_train[i]

x1 = tf.placeholder("float", [None, 1])   # 一次传入100个点[100,1]
W = tf.Variable(tf.zeros([1, 10]))
b = tf.Variable(tf.zeros([10]))

y1 = tf.matmul(x1, W)+b
z1 = tf.nn.sigmoid(y1)

'''

A = tf.constant([[1,0,2], [1,0,1],[0,0,1],[2,1,0]], dtype=tf.float32)
B = tf.constant([[1,0,2], [1,0,1],[0,0,1],[2,1,0]], dtype=tf.float32)
C = tf.constant([[1,0,2]], dtype=tf.float32)

with tf.Session() as sess:
    print(sess.run(init_op))

    # print(sess.run(y1, feed_dict={x1: x_t}))
    # print(sess.run(z1, feed_dict={x1: x_t}))
    print(sess.run(tf.multiply(A*B,C)))
