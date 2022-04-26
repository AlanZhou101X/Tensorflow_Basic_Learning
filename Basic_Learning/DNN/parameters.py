# coding: utf-8
'''
定义一个专门存放参数的类
'''
import sys


def get_params(argv):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        epochs=50,
        batch_size=200,
        N=30000,  # 选取部分MNIST数据进行实验
        N_train=20000,
        N_validation=4000,
        dim_hidden=200,
        lr=0.01,
        cnt_hidden=3  # 隐藏层个数
    )
    # params['patience'] = int(0.1 * params['nb_epochs'])  # Stop training if patience reached
    # patience: 在监测质量经过多少轮次没有进度时即停止
    # ########### User defined parameters ##############
    if argv == '1':
        params['epochs'] = 50

    elif argv == '2':
        params['epochs'] = 2

    else:
        print('ERROR: unknown argument {}'.format(argv))
        sys.exit(0)

    for key, value in params.items():
        print("{}: {}".format(key, value))
    return params