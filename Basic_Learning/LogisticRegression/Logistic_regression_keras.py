# coding:utf-8
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential([
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
])
'''
或：
model = Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Activation('sigmoid'))
'''
# 随机梯度下降法
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
# 或门正确的输入输出数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])
model.fit(X, Y, epochs=200, batch_size=1)
# 检查分类的结果以及输出的概率
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)
print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)