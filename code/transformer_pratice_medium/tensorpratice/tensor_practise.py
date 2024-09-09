'''
Author: Jean_Leung
Date: 2024-09-02 11:00:52
LastEditors: Jean_Leung
LastEditTime: 2024-09-05 21:40:57
FilePath: \LinearModel\tensorpratice\tensor_practise.py
Description: CNN 张量理解

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''

import torch
from torch import nn

# # 随机二维张量
# x = torch.randn(3,4)
# print("x: ", x)

# # 全零二维张量
# x = torch.zeros(4,3,dtype=torch.long)
# print("x: ", x)

# # 全一二维张量
# x = torch.ones(4,3,dtype=torch.float32)
# print("x :",x)

# # 通过tensor构建
# x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print("x: ", x)

# # 基于之前存在tensor构建新tensor
# x = torch.ones(4,4,dtype=torch.float32)
# print("x: ", x)
# # 基于该张量的size重新搞一个随机张量
# x = torch.rand_like(x,dtype=torch.double)
# print("x: ", x)
# print("x.shape:",x.shape)
# print("x.size:",x.size())

# # 使用张量arrange设定步长
# x = torch.arange(0,10,2) # arange是左闭右开
# print("x: ", x)
# print("x.shape",x.shape)
# print("x.size:",x.size())


# 使用normal设定
x = torch.normal(mean=0,std=0.1,size=(4,4))
print("x: ", x)

x = torch.normal(mean=torch.arange(4.),std=torch.arange(1.,0.6,-0.1)).reshape(2,2)
print("x: ", x)


y = nn.Tanh()
