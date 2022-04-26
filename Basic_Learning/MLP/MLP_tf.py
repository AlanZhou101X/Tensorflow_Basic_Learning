# coding: utf-8
'''
异或门XOR
不能用一条直线分类：线性不可分
简单感知机、逻辑回归都是只支持线性可分问题，这种模型被称为线性分类器
可以用组合基本门的思想来解决
研究误差的逆向输出来计算梯度的算法叫反向传播算法
始终记住，需要求的是可以使误差函数最小化的梯度
'''
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.truncated_normal([2, 2]))
# truncated_normal()生成遵循截断正态分布数据的方法
# 如果用tf.zeros()使初始化参数均为0，可能会导致反向传播算法无法正确反馈误差
b = tf.Variable(tf.zeros([2]))
h = tf.nn.sigmoid(tf.matmul(x, W)+b)
# 用同样的思路编写隐藏层-输出层代码
V = tf.Variable(tf.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)
# 设置训练时的误差函数，二分类可以使用交叉熵误差函数
cross_entropy = -tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 由于迭代次数较多，所以输出一下进度
for epoch in range(4000):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })
    if epoch % 1000 == 0:
        print("epoch:", epoch)

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess, feed_dict={
    x: X
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
print('w: ', sess.run(W))
print('b: ', sess.run(b))

