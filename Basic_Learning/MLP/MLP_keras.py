import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.utils import shuffle

# 加载数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()

# 输入层->隐藏层
model.add(Dense(input_dim=2, units=2))
# 或：model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))

# 隐藏层->输出层
model.add(Dense(units=1))
# 或：model.add(Dense(1))
model.add(Activation('sigmoid'))
# 随机梯度下降法求梯度
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01))
# 开始训练
model.fit(X, Y, epochs=4000, batch_size=4)
'''
或：
classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=1)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
'''

