# coding:utf-8

import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

mnist = datasets.fetch_mldata('MNIST original', data_home='./datasets')
# mnist = fetch_openml("mnist_784")
n = len(mnist.data)
N = 1000       # 选取部分mnist数据集进行实验
# 随机选取N个数据
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
r = mnist.target[indices]
# 转换为one-hot独热编码形式
Y = np.eye(10)[r.astype(int)]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

n_in = len(X[0])   # 输入维度784
n_hidden = 200      # 隐藏层维度200
n_out = len(Y[0])    # 输出维度10

'''
定义lrelu函数
'''
def lrelu(x, alpha=0.01):
    return tf.maximum(alpha*x, x)

def prelu(x, alpha):
    return tf.maximum(tf.zeros(tf.shape(x)), x) \
           + alpha * tf.maximum(tf.zeros(tf.shape(x)), x)

x = tf.placeholder(tf.float32, shape=[None, n_in])
t = tf.placeholder(tf.float32, shape=[None, n_out])
keep_prob =tf.placeholder(tf.float32)

W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
# stddev表示正态分布在截断前的标准差。
b = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.sigmoid(tf.matmul(x, W)+b)
h0_drop = tf.nn.dropout(h, keep_prob)

'''
增加隐藏层
'''
W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden]))
b1 = tf.Variable(tf.zeros([n_hidden]))
h1 = tf.nn.sigmoid(tf.matmul(h, W1)+b1)
h1_drop = tf.nn.dropout(h1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden]))
b2 = tf.Variable(tf.zeros([n_hidden]))
h2 = tf.nn.sigmoid(tf.matmul(h1, W2)+b2)
h2_drop = tf.nn.dropout(h2, keep_prob)

# 用同样的思路编写隐藏层-输出层代码
V = tf.Variable(tf.truncated_normal([n_hidden, n_out]))
c = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h2, V) + c)

# cross_entropy = -tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
batch_size = 100
n_batches = N // batch_size

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 由于迭代次数较多，所以输出一下进度
for epoch in range(1000):
    X_ , Y_ = shuffle(X_train, Y_train)
    for i in range(n_batches):
        start = i*batch_size
        end = start+batch_size

# 实际训练时需要加dropout
    sess.run(train_step, feed_dict={
        x: X_[start: end],
        t: Y_[start: end],
        keep_prob: 0.5
    })

    if epoch % 100 == 0:
        print("epoch:", epoch)

# 训练后的测试阶段不进行dropout
accuracy_rate = accuracy.eval(session=sess, feed_dict={
    x: X_test,
    t: Y_test,
    keep_prob: 1.0
})
print('accuracy:',  accuracy_rate)
