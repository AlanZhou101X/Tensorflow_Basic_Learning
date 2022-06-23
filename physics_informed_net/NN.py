# coding: utf-8
'''
定义模型输出->定义误差函数->训练模型
一般按照以下模式进行训练
def inference(x):
    # 定义模型

def loss(y, t):
    # 定义误差函数

def training(loss):
    # 定义训练算法

if __name__ == '__main__':
    # 1、准备数据
    # 2、设置模型
    y = inference(x)  #（1）定义模型
    loss = loss(y, t)   # （2）定义误差函数
    train_step = training(loss) # 定义训练算法
    # 3、训练模型
    # 4、评估模型
'''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys
import parameters
# from sympy import diff
# from sympy import symbols
import math


def inference(z, n_in, n_hiddens, n_out):
    def weight_variable(shape):  # 定义权重
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):    # 定义偏置
        initial = tf.zeros(shape)
        return tf.Variable(initial)
    output = None

    # 输入层->隐藏层， 隐藏层->隐藏层
    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = z
            input_dim = n_in
        else:
            input = output
            input_dim = n_hidden[i-1]

        W = weight_variable([input_dim, n_hidden])
        b = bias_variable([n_hidden])

        h = tf.nn.sigmoid(tf.matmul(input, W) + b)
        # output = tf.nn.dropout(h, keep_prob)
        # print(tmp.shape)
        # print(h.shape)
        tmp = h

    # 隐藏层->输出层
    W_out = weight_variable([n_hiddens[-1], n_out])
    b_out = bias_variable([n_out])
    # y = tf.nn.softmax(tf.matmul(h, W_out)+b_out)
    y = tf.matmul(tmp, W_out) + b_out
    # print(y.shape)
    return y, tmp, W, W_out


def training(loss, lr):   # 开始训练
    optimizer = tf.train.AdamOptimizer(lr)
    # 使用SGD优化器，设置初始学习率为lr
    train_step = optimizer.minimize(loss)
    return train_step


def get_accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def func(km,z):
    # 遍历z，对每个元素进行sin()操作
    siz_z = z.copy()
    for i in range(len(z)):
        siz_z[i] = math.sin(km*z[i])
    return siz_z


def get_loss(y, k, t, w, v, H):
    # 直接在损失函数里面加上边界条件加以限制，这样做明显简单了许多
    '''
  dif = tf.matmul(tf.multiply(y1*(1-y1), W), W1)  # dy/dx,dif形状[100,1],即对应点的导数值
  t_loss = (dif+(x1+lq)*y-x1**3-2*x1-lq*x1*x1)**2   # 常微分方程F的平方
  loss = tf.reduce_mean(t_loss)+(y[0]-1)**2   # 每点F平方求和后取平均再加上边界条件
    '''

    # loss = diff(fun(z), z, 2) + (k**2) * fun(z)
    dif1 = tf.matmul(tf.multiply(t*(1-t), w), v)
    dif2 = tf.matmul(tf.multiply(t*(1-t)*(1-2*t), w**2), v)  # 二阶导
    t_loss = (dif2 + k**2*y)**2     # 常微分方程F的平方
    loss = tf.reduce_mean(t_loss)+(y[0]-0)**2+(dif1[H]-1)**2       # 每点F平方求和后取平均再加上边界条件
    return loss, t_loss


def main(argv):
    '''
    数据的生成

    # 随机种子
    np.random.seed(0)
    tf.set_random_seed(1234)
    '''

    params = parameters.get_params(argv)
    km = params['km']
    k0 = params['k0']
    n = params['n']
    H = params['H']
    c = params['c']
    f = params['f']
    k0 = (2*math.pi*f)/c
    km = ((n-1/2)*math.pi)/H  
    p_cnt = params['p_cnt']
    x_train = np.linspace(0, H, p_cnt, endpoint=True)  # 在[0,H]等距采样p_cnt个点
    y_trail = func(km, x_train)                           # 已知解析式作为参照标签
    x_t = np.zeros((len(x_train), params['out_dim']))                # 新建一个len(x_train)*1的矩阵x_t
    for i in range(len(x_train)):                    # 将采样得到的点放入该矩阵
        x_t[i] = x_train[i]
    # tmp = np.zeros((len(x_train), 1))
    '''
    模型的设置
    '''
    n_in = len(x_t[0])
    cnt_hidden = params['cnt_hidden']
    n_hiddens = [params['dim_hidden']] * cnt_hidden  # 定义数组存放各隐藏层的维度
    n_out = params['out_dim']
    # p_keep = 0.5

    z = tf.placeholder(tf.float32, shape=[None, n_in])
    # t = tf.placeholder(tf.float32, shape=[None, n_out])    # 实际值，ground truth
    #keep_prob = tf.placeholder(tf.float32)

    y, y_tmp, w, v = inference(z, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)  # 预测值

    
    ls, t_loss = get_loss(y, km, y_tmp, w, v, H)
    lr = params['lr']
    train_step = training(ls, lr)
    

    # accuracy = get_accuracy(y, t)
    #
    # history = {
    #     'val_loss': [],
    #     'val_acc': []
    # }

    '''
    模型的训练
    '''
    nb_epochs = params['epochs']
    # batch_size = params['batch_size']

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(nb_epochs):  # 训练epochs次
        sess.run(train_step, feed_dict={z: x_t})
        if i % 50 == 0:
            total_loss = sess.run(ls, feed_dict={z: x_t})
            print("epoch:{},loss={}".format(i,total_loss))
            # print(sess.run(y[0], feed_dict={z: x_t}))
    saver = tf.train.Saver(max_to_keep=1)  # 保存模型，训练一次后可以将训练过程注释掉
    saver.save(sess, 'ckpt2/nn.ckpt', global_step=nb_epochs)
    saver = tf.train.Saver(max_to_keep=1)
    model_file = "ckpt2/nn.ckpt-"+str(nb_epochs)
    saver.restore(sess, model_file)
    output = sess.run(y, feed_dict={z: x_t})
    output1 = sess.run(t_loss, feed_dict={z: x_t})
    y_output = x_train.copy()
    # print(output.shape)
    # print(output1.shape)
    # print(y_output.shape)
    y_output1 = x_train.copy()

    # print(x_train)
    
    for i in range(len(x_train)):
        y_output[i] = output[i]
        y_output1[i] = output1[i]
        
    plt.figure()           # 图1画实际值与预测值的对比
    plt.title("Contrast of Ground truth and Prediction", loc='center')
    l1 = y_trail
    l2 = y_output
    # plt.plot(x_train, l1, label='ground truth')
    plt.plot(x_train, l2, label='prediction')
    plt.legend(loc='best', frameon=True)
    plt.savefig("Contrast.jpg")
    
    # plt.figure()           # 图2画实际值与预测值得偏差
    # plt.title("The bias of Ground truth and Prediction", loc='center')
    # plt.plot(x_train, y_trail - y_output)
    # plt.savefig("Bias.jpg")
    # plt.figure()           # 图3画出每一点对Loss的贡献
    # plt.title("View of Loss", loc='center')
    # plt.plot(x_train, y_output1 + (y_output[0] - 0) ** 2)
    # plt.savefig("Loss.jpg")
    # plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main('4')
