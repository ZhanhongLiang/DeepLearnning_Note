'''
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑     永不宕机     永无BUG
'''
'''
Author: Jean_Leung
Date: 2024-08-21 13:01:07
LastEditors: Jean_Leung
LastEditTime: 2024-08-21 13:01:19
FilePath: \LinearModel\net\multilayer_perceptron.py
Description: 多层感知机从零开始,自定义多层感知机

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
# 加载数据集合
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义图片输入参数, 每张图片都是28*28的灰度像素组成，所有图片分为10个类别
# 隐藏层单元有256个,通常定义2的幂次方
num_inputs, num_outputs, num_hiddens, num_hiddens2 = 784, 10, 256,64
# 定义W1的维度为 784 * 256
W1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True) * 0.01)
# 定义b1的维度为 256 
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
# 定义W1的维度为 256*10
W2 = nn.Parameter(torch.randn(num_hiddens,num_hiddens2,requires_grad=True) * 0.01)

W22 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True) * 0.01)
# 定义b2的维度为 64
b2 = nn.Parameter(torch.zeros(num_hiddens2,requires_grad=True))
# 定义b2的维度为 10
b22 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

W3 = nn.Parameter(torch.randn(num_hiddens2,num_outputs,requires_grad=True) * 0.01)

b3 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

# 定义多层感知机
params = [W1,b1,W2,b2,W3,b3]
params2 = [W1,b1,W22,b22]

# 定义激活函数RELU函数
def relu(X):
    a = torch.zeros_like(X) # 返回一个和X同个维度的0矩阵
    return torch.max(X,a)

# 定义最简单单层隐藏层模型
def net1(X):
    X = X.reshape(-1, num_inputs) # 展平每张图片
    H1 = relu(torch.matmul(X, W1) + b1) # 隐藏层1
    H2 = relu(torch.matmul(H1, W2) + b2) # 隐藏层2
    return torch.matmul(H2, W3) + b3 # 输出层

def net2(X):
    X = X.reshape(-1, num_inputs) # 展平每张图片
    H = relu(torch.matmul(X, W1) + b1) # 隐藏层1
    return torch.matmul(H, W22) + b22 # 输出层

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.3
updater = torch.optim.SGD(params, lr=lr)
updater2 = torch.optim.SGD(params2, lr=lr)
d2l.train_ch3(net2, train_iter, test_iter, loss, num_epochs, updater2)
d2l.plt.show()