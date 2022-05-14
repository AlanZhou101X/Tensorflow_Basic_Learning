import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x_train = np.linspace(0, 2, 100, endpoint=True)   # 生成[0,2]区间100个点
y_trail = np.exp(-0.5*x_train**2)/(1+x_train+x_train**3)+x_train**2   # 已知解析解用于比较
x_t = np.zeros((len(x_train), 1))
for i in range(len(x_train)):
    x_t[i] = x_train[i]

x1 = tf.placeholder("float", [None, 1])   # 一次传入100个点[100,1]
W = tf.Variable(tf.zeros([1, 10]))    # 输入到神经元的权重W
b = tf.Variable(tf.zeros([10]))
y1 = tf.nn.sigmoid(tf.matmul(x1, W)+b)   # sigmoid激活函数y1的形状[100,10]
W1 = tf.Variable(tf.zeros([10, 1]))      # 激活函数到输出的权重v
b1 = tf.Variable(tf.zeros([1]))
y = tf.matmul(y1, W1)          # 网络的输出[100,1]
lq = (1+3*(x1**2))/(1+x1+x1**3)
dif = tf.matmul(tf.multiply(y1*(1-y1), W), W1)  # dy/dx,dif形状[100,1],即对应点的导数值
t_loss = (dif+(x1+lq)*y-x1**3-2*x1-lq*x1*x1)**2   # 常微分方程F的平方
loss = tf.reduce_mean(t_loss)+(y[0]-1)**2   # 每点F平方求和后取平均再加上边界条件
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)   # Adam优化器训练网络参数
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(50000):    # 训练50000次
    sess.run(train_step, feed_dict={x1: x_t})
    if i % 50 == 0:
        total_loss = sess.run(loss, feed_dict={x1: x_t})
        print("loss={}".format(total_loss))
        print(sess.run(y[0], feed_dict={x1: x_t}))
saver = tf.train.Saver(max_to_keep=1)     # 保存模型，训练一次后可以将训练过程注释掉
saver.save(sess, 'ckpt/nn.ckpt', global_step=50000)
saver = tf.train.Saver(max_to_keep=1)
model_file = "ckpt/nn.ckpt-50000"
saver.restore(sess, model_file)
output = sess.run(y, feed_dict={x1: x_t})
output1 = sess.run(t_loss, feed_dict={x1: x_t})
y_output = x_train.copy()
y_output1 = x_train.copy()
for i in range(len(x_train)):
    y_output[i] = output[i]
    y_output1[i] = output1[i]
fig = plt.figure("预测曲线与实际曲线")
plt.plot(x_train, y_trail)
plt.plot(x_train, y_output)
fig2 = plt.figure("y_-y")  # 画实际值与预测值得偏差
plt.plot(x_train, y_trail-y_output)
fig3 = plt.figure("loss")  # 画出每一点对Loss的贡献
plt.plot(x_train, y_output1+(y_output[0]-1)**2)
plt.show()

