# coding:utf-8
'''
深度学习框架分为静态计算图和动态计算图。静态计算图的框架如Tensorflow、Caffe等，
动态计算图有PyTorch、Chainer等。静态如执行效率高，但灵活性较差，动态图牺牲效率保证灵活性，调试较方便
'''
import numpy as np

rng = np.random.RandomState(123)
# 控制随机数的状态，为方便观测试验结果，得到多个遵循正态分布的数据，每次需生成“同样的”随机数据

d = 2   # 数据维度
N = 10   # 每组数据的数量
mean = 5  # 神经元激活的那一组数据的平均值

x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

# 为一次性处理生成这两种数据，把x1和x2数据合并到一起
x = np.concatenate((x1, x2), axis=0)

# 接下来用感知机对生成的数据分类。首先对模型的参数权重向量w和偏置b初始化
w = np.zeros(d)
b = 0
# 然后用函数定义输出y=f(wTx+b)


def y(x):
    return step(np.dot(w, x)+b)


def step(x):          # x是阶跃函数
    return 1*(x > 0)


# 需要正确的输出值更新参数，因此要像下面的定义输出
def t(i):
    if i < N:           # 前N个数据是不激活的x1，剩余的N个数据是激活的x2
        return 0
    else:
        return 1


# 训练所需的函数值准备就绪，下面开始误差修正学习法——不断重复训练直到所有数据被正确分类
while True:
    classified = True
    for i in range(N*2):
        delta_w = ((t(i) - y(x[i]))*x[i])
        delta_b = (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w == 0)*(delta_b == 0)
    if classified:
        break
'''
while True:
    #
    #  参数更新的部分（需要实现w、b的更新表达式，判断所有数据是否被正确跟累的逻辑）
    #
    if ‘所有数据都被正确分类’:
        break
'''
classified *= all(delta_w == 0)*(delta_b == 0)
print(y([0, 0]))
print(y([5, 5]))





