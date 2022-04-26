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
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys
import parameters


def inference(x, keep_prob, n_in, n_hiddens, n_out):
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
            input = x
            input_dim = n_in
        else:
            input = output
            input_dim = n_hidden[i-1]

        W = weight_variable([input_dim, n_hidden])
        b = bias_variable([n_hidden])

        h = tf.nn.tanh(tf.matmul(input, W) + b)
        output = tf.nn.dropout(h, keep_prob)

    # 隐藏层->输出层
    W_out = weight_variable([n_hiddens[-1], n_out])
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(output, W_out)+b_out)
    return y


def training(loss, lr):   # 开始训练
    optimizer = tf.train.GradientDescentOptimizer(lr)
    # 使用SGD优化器，设置初始学习率为lr
    train_step = optimizer.minimize(loss)
    return train_step


def get_accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def get_loss(y, t):
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
            t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),  # 设置计算是要用的下限值
            reduction_indices=[1]))
    return cross_entropy


def main(argv):
    '''
    数据的生成
    '''
    # 随机种子
    np.random.seed(0)
    tf.set_random_seed(1234)

    mnist = datasets.fetch_mldata('MNIST original', data_home='./datasets')

    n = len(mnist.data)
    params = parameters.get_params(argv)
    N = params['N']  # 选取部分MNIST数据进行实验
    N_train = params['N_train']
    N_validation = params['N_validation']
    indices = np.random.permutation(range(n))[:N]  # 随机选择N个数据

    X = mnist.data[indices]
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 转换为1-of-K形式

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, train_size=N_train)

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X_train, Y_train, test_size=N_validation)

    '''
    模型的设置
    '''
    n_in = len(X[0])
    cnt_hidden = params['cnt_hidden']
    n_hiddens = [params['dim_hidden']] * cnt_hidden  # 定义数组存放各隐藏层的维度
    n_out = len(Y[0])
    p_keep = 0.5

    x = tf.placeholder(tf.float32, shape=[None, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    keep_prob = tf.placeholder(tf.float32)

    y = inference(x, keep_prob, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)
    loss = get_loss(y, t)
    lr = params['lr']
    train_step = training(loss, lr)

    accuracy = get_accuracy(y, t)

    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
    模型的训练
    '''
    epochs = params['epochs']
    batch_size = params['batch_size']

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                keep_prob: p_keep
            })

        # 使用验证数据进行评估
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })

        # 记录验证数据的训练进度
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print('epoch:', epoch,
              ' validation loss:', val_loss,
              ' validation accuracy:', val_acc)

    '''
    训练进度情况可视化
    '''
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax_acc = fig.add_subplot(111)
    ax_acc.plot(range(epochs), history['val_acc'],
                label='acc', color='black')
    ax_loss = ax_acc.twinx()
    ax_loss.plot(range(epochs), history['val_loss'],
                 label='loss', color='gray')
    plt.xlabel('epochs')
    # plt.show()
    plt.savefig('mnist_tensorflow.jpg')

    '''
    评估预测精度
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_test,
        t: Y_test,
        keep_prob: 1.0
    })
    print('accuracy: ', accuracy_rate)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main('1')
