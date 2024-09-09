'''
Author: Jean_Leung
Date: 2024-08-21 16:56:51
LastEditors: Jean_Leung
LastEditTime: 2024-08-22 16:52:41
FilePath: \LinearModel\net\multilayer_perceptron2.py
Description: 多层感知机使用框架

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''
import torch
from torch import nn
from d2l import torch as d2l
# 按照里面方法进行顺序执行
net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
net2 = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.Sigmoid(),nn.Linear(256,10))
net3 = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.Tanh(),nn.Linear(256,10))

# # 初始化权重,normal初始化方法
# def init_weights(m):
#     if type(m) == nn.Linear:
#         # nn.init.normal_(m.weight, std=0.01)
#         nn.init.xavier_uniform(m.weight)
#         nn.init.constant_(m.bias, 0)

# 也可以在init_weights函数中对每一层进行初始化
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # nn.init.normal_(m.weight,std=0.01)
            nn.init.xavier_uniform_(m.weight)
            # nn.init.constant_(m.bias, 0)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
# 添加模型中
net.apply(init_weights)


batch_size, lr, num_epochs = 256,0.1,10

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net3.parameters(),lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net3, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()

# 多个激活函数进行比较