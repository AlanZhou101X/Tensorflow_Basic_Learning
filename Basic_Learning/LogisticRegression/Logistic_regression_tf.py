# coding:utf-8
'''
问：为什么用sigmoid函数作为激活函数？
答：简单感知机神经元输出值只有0和1，但这样做预测判断未免太“粗暴”
因此考虑把输出值设置成概率的形式，用sigmoid函数代替阶跃函数，化离散为连续
sigmoid函数用作激活函数，一方面是因为输出范围在0到1之间，
另一方面，如果用正态分布的累积分布函数代替sigmoid函数成为激活函数，此模型称为“probit回归”，
但是若使用其累积分布函数进行神经元建模，当使用随机梯度下降法时需要进行必要的梯度运算，此过程繁琐困难，
因此使用和正态分布函数相似、计算过程简单地sigmoid函数作为激活函数
'''

'''
调整参数是似然函数最大化，就可以很好地训练网络，求函数最大或最小状态的问题称为最优化问题
“最优化”是指求使函数最小化的参数。
似然函数是N个输入数据对应的输出数据概率的乘积，将似然函数先取log后偏微分，化乘积为和的形式，
可得到交叉熵误差函数E(w,b)，E称为“误差函数”或“损失函数”
'''

'''
梯度下降法：
学习率作为超参数，可用于调整模型参数收敛的难度，一般采用0.1、0.01等更小的值
学习率过大或过小都不行，过大会导致求得的值在最优解上下变来变去，因此难以收敛
学习率过小会导致收敛到x*为止的迭代数会增加，且得到的可能是局部最优解，而难以得到真正的全局最优解x*,
因此通常采用学习率一开始较大，后来慢慢变小的方法
随机梯度下降(SGD)和小批量梯度下降：
由于梯度下降法需要先对N个数据求和，若N非常大，则数据无法一次性放入内存，增加计算时间，
（1）随机梯度下降：
因此随机梯度下降法每次选择一个数据更新参数，“随机”是指选择的顺序随机，而这种方式需要对N个数据反复训练，
称为“迭代”epoch
for epoch in range(epochs):
    shuffle(data)   # 每次迭代后打乱数据顺序
    for datum in data:   # 每次都使用一个数据更新参数
    parmas_grad = evaluate_gradient(error_function, params, datum)
    params -= learning_rate * params_grad
    但是SGD有不稳定性，可能导致算法收敛速度较低。
    现多用基于权重的二阶导数的优化器，如AdaGrad、Adam，引入了动量、权重梯度的平方
（2）小批量梯度下降：
把N个数据分成M个小块，再进行训练，M一般会选50~500
for epoch in range(epochs):
    shuffle(data)   # 每次迭代后打乱数据顺序
    batches = get_batches(data, batch_size=M)
    for batch in bathes:   # 每次都使用一个数据更新参数
    parmas_grad = evaluate_gradient(error_function, params, batch)
    params -= learning_rate * params_grad
'''

import numpy as np
import tensorflow as tf
# 训练或门时，输入是二维、输出是一维的
w = tf.Variable(tf.zeros([2, 1]))  # 生成变量
b = tf.Variable(tf.zeros([1]))

'''
# 定义模型输出表达式_原生python：
def y(x):
    return sigmoid(np.dot(w,x)+b)
    
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
'''


# Tensorflow定义输出表达式
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
# 定义模型时只需预先确定数据的维度，直到模型训练到实际需要的情况再代入值
y = tf.nn.sigmoid(tf.matmul(x, w)+b)
# 计算交叉熵 tf.reduce_sum()相当于np.sum()
cross_entropy = -tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))
# 确定表达式时对交叉熵函数的各个参数偏微分、求梯度
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# 编写检查训练结果是否正确的代码
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 模型设置完毕，下面是实际训练部分，首先定义用于训练的数据
# 或门
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])
# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 训练
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })
# 检查训练结果,feed_dict可以向作为placeholder的x和t代入实际的值
# 可以用.eval()来确认是否被激活、有没有被正确分类等问题
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
print('w: ', sess.run(w))
print('b: ', sess.run(b))
'''
流程：定义模型-》定义误差函数-》定义最优化方法-》会话初始化-》训练
'''


