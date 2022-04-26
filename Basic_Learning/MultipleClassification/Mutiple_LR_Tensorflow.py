# 多分类问题，需要softmax函数
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
M = 2  # 输入数据的维度
K = 3  # 分类数
n = 100  # 每个数据的分类个数
N = n*K   # 全部数据的个数
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 为了求得每个小批量的平均值，用tf.reduce_mean。reduction_indices表示延哪个方向计算和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y), reduction_indices=[1]))
# 用随机梯度下降法对交叉熵误差函数最小化
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# 检查分类是否正确
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
batch_size = 50
n_batches = N // batch_size
# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(20):
    X_, Y_ = shuffle(X, Y)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        '''
        init = tf.global_variables_initializer()
        sess = tf.Session()
        '''
        sess.run(train_step, feed_dict={
            x: X_[start: end],
            t: Y_[start: end]
        })

X_, Y_ = shuffle(X, Y)

# 先选取十个，看看是否已正确分类
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0: 10],
    t: Y_[0: 10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0: 10]
})
print('classified:')
print(classified)
print()
print('output probability:')
print(prob)


