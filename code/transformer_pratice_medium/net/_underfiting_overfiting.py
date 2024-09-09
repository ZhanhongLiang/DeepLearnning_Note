'''
Author: Jean_Leung
Date: 2024-08-25 10:24:25
LastEditors: Jean_Leung
LastEditTime: 2024-09-01 10:03:50
FilePath: \LinearModel\net\_underfiting_overfiting.py
Description: 

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''


import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

#---------------------------第一步:定义数据集合--------------------------
# 定义多项式函数
max_degree = 20 # 多项式的最大阶数

n_train, n_test = 100, 100 # 训练和测试数据集⼤⼩

true_w = np.zeros(max_degree) # 分配⼤量的空间

true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# print("true_w:",true_w)
# 噪音
# [200,1] 代表随机生成的数据
# 代表x系数
features = np.random.normal(size=(n_train + n_test, 1)) # 200行,1列的矩阵
# print("features.shape:",features.shape)
# print("features:",features)

# 打乱随机生成的数据
np.random.shuffle(features) # shuffle 打乱数据

# np.arrange(max_degree).reshape(1,-1)就是生成一个0~20的一行，20列的行向量
# 通过np.power过后，poly_features变成[200,20]的数据
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) # features 和 np.arrange的列数要相同

b = np.arange(max_degree).reshape(1, -1) # features

print("b:",b)
print("b.shape= ", b.shape)

# print("poly_features:",poly_features)
# # poly_features的维度是
# print("poly_features.shape:",poly_features.shape)

for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # `gamma(n)` = (n-1)!

# `labels`的维度: (`n_train` + `n_test`,) 也就是(200,)
# np.dot对矩阵进行运算得到就是矩阵积
# 得到就是200个样本的
labels = np.dot(poly_features, true_w) # 系数与x项和阶乘项相乘，要用点成
print("labels",labels)
labels += np.random.normal(scale=0.1, size=labels.shape) # 与噪音项相加
print("labels.shape",labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

# print("features:",features[:2]) # 取出前两行出来
# # print("features.shape:",features.shape[-1])

# print("poly_features:",poly_features[:2]) # 取出前两行出来

# print("poly_features.shape:",poly_features.shape[-1])


# print("labels:",labels[:2]) # 取出前两行出来

#---------------------------第二步: 定义模型--------------------------

# 定义损失函数
def evaluate_loss(net, data_iter, loss): #@save
    """评估模型在数据集上的��失"""
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数,将初始化一个只有两个数的数组
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# n = d2l.Accumulator(2)
# print("n = ", n.data)

# 定义训练模型
# train_features是训练数据，test_features是测试数据，train_labels是测试层数
def train(train_features, test_features, train_labels, test_labels,num_epochs=400):
    loss = nn.MSELoss() # 计算均方误差 
    input_shape = train_features.shape[-1] # 得到train_features的列数,这里列数为20
    # 定义模型
    # 不设置偏置，因为我们已经在多项式特征中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 通过序列模型
    batch_size = min(10, train_labels.shape[0]) # batch_size是分批样本数
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),batch_size) # 训练数据集合，
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),batch_size, is_train=False) # 测试集合加载数据
    trainer = torch.optim.SGD(net.parameters(), lr=0.01) # SGD通过随机梯度下降算法进行梯度下降
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs], ylim=[1e-3, 1e2],legend=['train', 'test']) # 画图算法
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer) # 通过训练，这个调用之前所有的函数
        if epoch == 0 or (epoch + 1) % 20 == 0: # epoch是一轮训练
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# # 从多项式特征中选择前4个维度，即 1, x, x^2/2!, x^3/3!
# # 正态，三阶多项式函数拟合
# train(poly_features[:n_train, :4], poly_features[n_train:, :4],labels[:n_train], labels[n_train:])
# d2l.plt.show()

# # 从多项式特征中选择前2个维度，即 1, x
# train(poly_features[:n_train, :2], poly_features[n_train:, :2],labels[:n_train], labels[n_train:])
# d2l.plt.show()

# 从多项式特征中选取所有维度
print("poly_features[:n_train]",poly_features[:n_train,:])
train(poly_features[:n_train,:],poly_features[n_train:,:],labels[:n_train], labels[n_train:],num_epochs=1500)
d2l.plt.show()


