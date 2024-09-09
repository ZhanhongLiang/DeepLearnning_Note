'''
Author: Jean_Leung
Date: 2024-08-20 22:37:05
LastEditors: Jean_Leung
LastEditTime: 2024-09-05 21:43:24
FilePath: \LinearModel\net\activation_function.py
Description: 

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# alpha = torch.range(0.025,0.025,0.1,requires_grad=True)
# # print("alpha: ",alpha) # 打印出来的是张量tensor类型
# # alpha = torch.arange()
# y = torch.prelu(x,alpha)
# d2l.plot(x.detach(),y.detach(),x.grad,'x','prelu(x)',figsize=(5,2.5))
# d2l.plt.show()

# # 接下来是prelu导函数
# y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(),x.grad,'x','grad of prelu(x)',figsize=(5,2.5))
# d2l.plt.show()

# 接下来是tanh导函数
# y = torch.tanh(x) + 1
# d2l.plt.subplot(1,2,1)
# d2l.plot(x.detach(),y.detach(),x.grad,'x','tanh(x)',figsize=(5,2.5))
# d2l.plt.title('tanh(x) + 1')
# y = 2 * torch.sigmoid(2 * x)
# d2l.plt.subplot(1,2,2)
# d2l.plot(x.detach(),y.detach(),x.grad,'x','2 * sigmoid(2x)',figsize=(5,2.5))
# d2l.plt.title('2 * sigmoid(2x)')
# d2l.plt.show()

y = torch.tanh(x)
d2l.plt.subplot(1,2,1)
d2l.plot(x.detach(),y.detach(),x.grad,'x','tanh(x)',figsize=(5,2.5))
d2l.plt.title('tanh(x)')

y = 1- torch.tanh(x) * torch.tanh(x)

d2l.plt.subplot(1,2,2)
d2l.plot(x.detach(),y.detach(),x.grad,'x','f''(x)',figsize=(5,2.5))
d2l.plt.title('f''(x)')
d2l.plt.show()