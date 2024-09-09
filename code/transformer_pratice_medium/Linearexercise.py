'''
......................................&&.........................
....................................&&&..........................
.................................&&&&............................
...............................&&&&..............................
.............................&&&&&&..............................
...........................&&&&&&....&&&..&&&&&&&&&&&&&&&........
..................&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&..............
................&...&&&&&&&&&&&&&&&&&&&&&&&&&&&&.................
.......................&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&.........
...................&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&...............
..................&&&   &&&&&&&&&&&&&&&&&&&&&&&&&&&&&............
...............&&&&&@  &&&&&&&&&&..&&&&&&&&&&&&&&&&&&&...........
..............&&&&&&&&&&&&&&&.&&....&&&&&&&&&&&&&..&&&&&.........
..........&&&&&&&&&&&&&&&&&&...&.....&&&&&&&&&&&&&...&&&&........
........&&&&&&&&&&&&&&&&&&&.........&&&&&&&&&&&&&&&....&&&.......
.......&&&&&&&&.....................&&&&&&&&&&&&&&&&.....&&......
........&&&&&.....................&&&&&&&&&&&&&&&&&&.............
..........&...................&&&&&&&&&&&&&&&&&&&&&&&............
................&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&............
..................&&&&&&&&&&&&&&&&&&&&&&&&&&&&..&&&&&............
..............&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&....&&&&&............
...........&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&......&&&&............
.........&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&.........&&&&............
.......&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&...........&&&&............
......&&&&&&&&&&&&&&&&&&&...&&&&&&...............&&&.............
.....&&&&&&&&&&&&&&&&............................&&..............
....&&&&&&&&&&&&&&&.................&&...........................
...&&&&&&&&&&&&&&&.....................&&&&......................
...&&&&&&&&&&.&&&........................&&&&&...................
..&&&&&&&&&&&..&&..........................&&&&&&&...............
..&&&&&&&&&&&&...&............&&&.....&&&&...&&&&&&&.............
..&&&&&&&&&&&&&.................&&&.....&&&&&&&&&&&&&&...........
..&&&&&&&&&&&&&&&&..............&&&&&&&&&&&&&&&&&&&&&&&&.........
..&&.&&&&&&&&&&&&&&&&&.........&&&&&&&&&&&&&&&&&&&&&&&&&&&.......
...&&..&&&&&&&&&&&&.........&&&&&&&&&&&&&&&&...&&&&&&&&&&&&......
....&..&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&...........&&&&&&&&.....
.......&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&..............&&&&&&&....
.......&&&&&.&&&&&&&&&&&&&&&&&&..&&&&&&&&...&..........&&&&&&....
........&&&.....&&&&&&&&&&&&&.....&&&&&&&&&&...........&..&&&&...
.......&&&........&&&.&&&&&&&&&.....&&&&&.................&&&&...
.......&&&...............&&&&&&&.......&&&&&&&&............&&&...
........&&...................&&&&&&.........................&&&..
.........&.....................&&&&........................&&....
...............................&&&.......................&&......
................................&&......................&&.......
.................................&&..............................
..................................&..............................
'''

'''
Author: Jean_Leung
Date: 2024-08-07 16:45:46
LastEditors: Jean_Leung
LastEditTime: 2024-08-07 16:47:45
FilePath: \LinearModel\Linearexercise.py
Description: 李沐学ai 线性回归、优化算法学习

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''


import random
import torch
from d2l import torch as d2l


# 定义一个函数来生成模拟数据集
# 生成数据集

def synthetic_data(w,b,num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0,1,(num_examples,len(w))) # 均值为0，标准差为1,且size为(num_examples,len(w))的X张量
    y = torch.matmul(X,w) + b # torch.matmul()是矩阵乘法 利用广播机制
    print("y.shape:",y.shape)
    y += torch.normal(0,0.01,y.shape)  # 加入高斯噪声 # 均值为0，标准差为0.01，size为y.shape的向量
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4]) # 噪音
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000) # X的size为(1000,2), y的size为(1000,1)

# print("features:",features[0],"labels:",labels[0])
# print("features:",features[:,1])
d2l.set_figsize() #  Set the figure size for matplotlib.
# # scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=<deprecated parameter>, edgecolors=None, *, plotnonfinite=False, data=None, **kwargs)
# x 为x轴，y为y轴，只有detach后才能转到numpy里面去
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1) # 将features的第二列所有数据先detach()后numpy化, labels的数据同样操作 
d2l.plt.show()


# print("features.size",features.shape)
# print("labels.size",labels.shape)

# 读取小批量
# 随机批量梯度算法

# data_iter函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
# 生成batch_size大小的随机小批量数据
def data_iter(batch_size, features, labels):
    # 先将序列号确定
    # indices = torch.randperm(features.shape[0]) # 随机将0到features.shape[0] - 1的数打乱
    num_examples = len(features) # 样本个数
    indices = list(range(num_examples)) # 0到nnum_examples - 1的数
    # print(indices[:]) # 从0~999的索引号
    random.shuffle(indices) # 随机打乱 indices
    # print(indices[:])
    # 然后将打乱的序列号分为多个batch
    for i in range(0,num_examples, batch_size):
        # print("i:",i)
        # indices里面索引号还是0~999，但是索引号对应的空间放的是随机数0~999
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)]) # 当i+batch_size超出时，取num_examples
        # print(batch_indices)
        yield features[batch_indices], labels[batch_indices] # yield可以暂时理解为return，但是下次继续执行该函数时候，从未完成开始

batch_size = 10

# 读取并打印第一个小批量
for X, y in data_iter(batch_size, features, labels):
    print("X:", X, "\ny:", y)
    break


# 定义初始化模型参数
# w初始化为均值0，标准差0.01，大小为(2,1)的列向量,可以放置梯度
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
# b初始化为0的标量，也可以放置梯度
b = torch.zeros(1,requires_grad=True)

# 完整训练模型
# 损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2 # 将y统一成与y_hat一样同尺寸 

# 线性回归方程组
# 定义模型
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b

# 定义优化w的算法
# sgd是 随机小批量梯度下降
def sgd(params,lr,batch_size):
    """小批量随即梯度下降"""
    with torch.no_grad(): # 不产生梯度计算，减少内存消耗
            for param in params: # 遍历每个参数
                param -= lr * param.grad / batch_size # 将进行梯度下降
                param.grad.zero_() # 进行梯度清零，方便下次计算，不让这次梯度影响下次梯度


# 定义训练函数
def train(features, labels, batch_size, lr, num_epochs):
    """线性回归模型的训练"""
    # w,b = w.clone().detach(), b.clone().detach() # 克��w和b，并detach()，让w和b不参与��度计算
    # print("w.grad:",w.grad)
    # print("b.grad:",b.grad)
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # linreg是y_hat的值，也就是预估值，y是真实值，y从data_iter中随机生成出来，模拟真实数据
            l = squared_loss(linreg(X, w, b), y) # 计算x和y的小批量损失
            # print(X[:])
            # 因为l是形状是(batch_size,1)，而不是一个标量。l中所有元素被加到一起
            # 并以此计算关于[w,b]的梯度
            l.sum().backward() # 反向传播
            sgd([w, b], lr, batch_size) # 随机小批量度下降，使用参数更新梯度
        with torch.no_grad(): # 这个是如果不计算梯度下降的时候
            # 这个不是随机小批量数据进行运算，直接1000个数据进行计算
            train_l = squared_loss(linreg(features,w,b),labels)
            # print(train_l[:])
            print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
            # print(f"epoch {epoch+1}, loss {l.item():f}") 
    # 比较真实参数和通过训练学到的参数来评估训练的成功程度
    # true_w是提前设定的w参数
    print(f'w的估计误差：{true_w-w.reshape(true_w.shape)}')
    print(f'b的估计误差：{true_b-b}')

# 开始训练

train(features, labels, batch_size, lr=0.03, num_epochs=3)

