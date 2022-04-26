# coding:utf=8
import tensorflow as tf
import numpy as np
from sklearn import datasets
# from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from sklearn.datasets.base import get_data_home

# print(get_data_home())

mnist = datasets.fetch_mldata('MNIST original', data_home='./datasets')
# mnist = fetch_openml("mnist_784")
n = len(mnist.data)
N = 1000       # 选取部分mnist数据集进行实验
# 随机选取N个数据
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
# 转换为one-hot独热编码形式
Y = np.eye(10)[y.astype(int)]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

'''
模型的设置
'''
n_in = len(X[0])   # 输入维度784
n_hidden = 200      # 隐藏层维度200
n_out = len(Y[0])    # 输出维度10

model = Sequential()
# 输入层->隐藏层1
model.add(Dense(n_hidden, input_dim=n_in))
# model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
# model.add(Activation('relu'))
# model.add(LeakyReLU())
# 隐藏层1->隐藏层2
model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
# model.add(Activation('relu'))
# model.add(LeakyReLU())
# 隐藏层2->隐藏层3
model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
# model.add(Activation('relu'))
# model.add(LeakyReLU())

# 隐藏层3->隐藏层4
model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
# model.add(Activation('relu'))
# model.add(LeakyReLU())

# 隐藏层->输出层
model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])


epochs = 1000
batch_size = 100
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

# 模型的评估
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
# 输出的第一个元素是误差函数的值，第二个元素是预测精度的值
'''
 增加神经元个数可以提升预测精度，但并不是越多越好，且需要注意计算的执行时间
 只简单增加隐藏层的个数也不能达到预期的效果，而且可能不增反降
 模型的层数越深，模型能够表现以及分类的模式应该是越多的，梯度消失问题越严重
 因为推导出的反向传播表达式中有一部分是sigmoid函数微分的乘积，如果有N层隐藏层，
 sigmoid函数的微分函数最大值只有0.25
 则计算误差时会乘上一个系数AN，因此如果隐藏层数量增加，误差项的值就会快速趋近于0
 梯度消失问题在层不深的情况下也可能出现，尤其是当每层的维度很多时
 双曲正切函数tanh(x)不易梯度消失，其导函数最大值为1，但在层数深、数据维度高时仍不起作用，值域是[-1,1]
 tanh(x) 提高了sigmoid函数的收敛速到
 ReLU函数梯度不会消失，且计算速度快，但x<=0时函数值和梯度都是0，因此可能会出现训练过程中一直不激活的现象
 Leaky ReLU和Parametric ReLU是ReLU的改进版，后者x<0部分是一个向量，需要最优化的参数
 过拟合问题需要通过droupout层来解决
'''








