'''
Author: Jean_Leung
Date: 2024-08-08 11:50:34
LastEditors: Jean_Leung
LastEditTime: 2024-09-02 10:58:33
FilePath: \LinearModel\Linearexercise2.py
Description: 

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn   

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w,true_b,1000) # 库函数生成人工数据集  
# 调用框架现有的API来读取数据

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays) # 定义一个TensorDataset来将数据和标签整合,一个星号*，表示对list解开入参
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 定义一个DataLoader来读取数据,返回的是从dataset中随机挑选出batch_size个样本出来

batch_size = 10
# 这一句需要着重研究一下，返回的是数据的迭代器
data_iter = load_array((features,labels),batch_size)# 返回的数据的迭代器

print(next(iter(data_iter))) # iter(data_iter) 是一个迭代器对象，next是取迭代器里面的元素

# 使用框架的预定好的层
# nn 是神经网络缩写
net = nn.Sequential(nn.Linear(2,1))

# 初始化参数
net[0].weight.data.normal_(0,0.01) # 使用正态分布替换掉weight变量里面的数据值
net[0].bias.data.fill_(0) # 偏差bias变量里面的值设置为0
print(net[0])

# 计算均方误差使用的是MSELoss类，也称为平方L2范数
loss = nn.MSELoss()   #L1是算术差，L2是平方差

# 实例化SGD实例,返回trainer对象
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

# 训练过程代码与从零开始所做的非常相似
num_epochs = 3

# 定义训练函数，在这个函数中进行模型的训练
def train(features,labels,num_epochs):
    for epoch in range(num_epochs):
        for X,y in data_iter: # 从DataLoader里面一次一次把所有数据拿出来
            l = loss(net(X),y) # net(X) 为计算出来的线性回归的预测值
            trainer.zero_grad() # 梯度清零
            l.backward()
            trainer.step() # SGD优化器优化模型
        l = loss(net(features),labels)
        print(f'epoch{epoch+1},loss{l:f}')

train(features,labels,num_epochs)

