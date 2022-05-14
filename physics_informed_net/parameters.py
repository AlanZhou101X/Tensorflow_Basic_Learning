# coding: utf-8
'''
定义一个专门存放参数的类
'''
import sys
import math


def get_params(argv):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        epochs=100000,
        # batch_size=150,
        # batch_size=25,
        # dim_hidden=32,
        dim_hidden=10,  # 每个隐藏层的神经元个数
        # cnt_hidden=6,
        cnt_hidden=1,   # 隐藏层个数
        out_dim=1,      # 输出层形状
        lr=0.0001,      # 学习率
        H= 2 * math.pi,    # 输入变量z的边界条件H的值
        p_cnt=100,    # 在[0,H]等距采样p_cnt个点
        k=1          # 常微分方程中的常量k
    )
    # params['patience'] = int(0.1 * params['nb_epochs'])  # Stop training if patience reached
    # patience: 在监测质量经过多少轮次没有进度时即停止
    # ########### User defined parameters ##############

    if argv == '1':
        params['epochs'] = 50000
    elif argv == '2':
        params['epochs'] = 2
    elif argv == '3':
        params['epochs'] = 50000
        params['p_cnt'] = 10000
        params['cnt_hidden'] = 6
    else:
        print('ERROR: unknown argument {}'.format(argv))
        sys.exit(0)

    for key, value in params.items():
        print("{}: {}".format(key, value))
    return params

