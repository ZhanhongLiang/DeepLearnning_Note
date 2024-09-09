
'''
Author: Jean_Leung
Date: 2024-08-09 11:26:05
LastEditors: Jean_Leung
LastEditTime: 2024-08-09 13:21:20
FilePath: \LinearModel\softmax\pictureexercise.py
Description: 

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''
'''
                  江城子 . 程序员之歌

              十年生死两茫茫，写程序，到天亮。
                  千行代码，Bug何处藏。
              纵使上线又怎样，朝令改，夕断肠。

              领导每天新想法，天天改，日日忙。
                  相顾无言，惟有泪千行。
              每晚灯火阑珊处，夜难寐，加班狂。

'''

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# SVG是一种无损格式 – 意味着它在压缩时不会丢失任何数据，可以呈现无限数量的颜色。
# SVG最常用于网络上的图形、徽标可供其他高分辨率屏幕上查看。
d2l.use_svg_display() # 使用svg显示图片，这样清晰度高一些

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root="../01_Data/01_DataSet_FashionMNIST",train=True,transform=trans,download=True)

mnist_test = torchvision.datasets.FashionMNIST(root="../01_Data/01_DataSet_FashionMNIST",train=False,transform=trans,download=True)

# print(len(mnist_train)) # 训练数据集长度

# print(len(mnist_test)) # 测试数据集长度

# # 定义读取小批量数据
# print(mnist_train[0][0].shape) # 黑白图片，所以channel为1。
# print(mnist_train[0][1]) # [0][0]表示第一个样本的图片信息，[0][1]表示该样本对应的标签值

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 读取并打印部分示例
# 显示数据集，也就是显示图片出来

def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    """在一行里显示多张图像"""
    # d2l.use_svg_display(False)
    figsize = (num_cols * scale, num_rows * scale) # 传进来的图像尺寸，scale 为放缩比例因子
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    print(_) # 显示figures，就是显示图片的数据轴
    print(axes) # axes 为构建的两行九列的画布
    axes = axes.flatten()
    print(axes) # axes 变成一维数据，因为axes被flatten了
    for i, (ax, img) in enumerate(zip(axes, images)):
        if(i < 1):
            print("i:" , i)
            print("ax, img:",ax,img)
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
            ax.set_title(titles[i])
        else:
            # PIL 图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train,batch_size=18))) # X，y 为仅抽取一次的18个样本的图片、以及对应的标签值
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
d2l.plt.show()


# 小批量数据集
batch_size = 256

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4
# 上一章的内容，需要用DataLoader来进行进行加载数据迭代器
# 我们使⽤内置的数据迭代器，而不是从零开始创建⼀个。回顾⼀下，在每次迭代中，数据加载器每次都会读取⼀小批量数据，
train_iter = data.DataLoader(mnist_train,batch_size, shuffle=True, num_workers=get_dataloader_workers())
timer = d2l.Timer() # 计时器对象实例化,开始计时
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f}sec') # 计时器停止时，停止与开始的时间间隔事件


# 最终的整合成一个函数
# 现在我们定义 load_data_fashion_mnist 函数，⽤于获取和读取Fashion-MNIST数据集。它返回训练集
# 和验证集的数据迭代器。此外，它还接受⼀个可选参数，⽤来将图像⼤小调整为另⼀种形状。
def load_data_fashion_mnist(batch_size,resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize)) # 如果有Resize参数传进来，就进行resize操作
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../01_Data/01_DataSet_FashionMNIST",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../01_Data/01_DataSet_FashionMNIST",train=False,transform=trans,download=True)
    train_iter = data.DataLoader(mnist_train,batch_size, shuffle=True, num_workers=get_dataloader_workers())
    test_iter = data.DataLoader(mnist_test,batch_size, shuffle=False, num_workers=get_dataloader_workers())
    return train_iter, test_iter


