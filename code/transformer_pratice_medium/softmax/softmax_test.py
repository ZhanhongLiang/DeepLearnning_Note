'''
Author: Jean_Leung
Date: 2024-08-09 13:28:17
LastEditors: Jean_Leung
LastEditTime: 2024-08-18 18:49:09
FilePath: \LinearModel\softmax\softmax_test.PY
Description: 测试Softmax回归(不使用框架，自定义函数测试)

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''


import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


# 最终的整合成一个函数
# 现在我们定义 load_data_fashion_mnist 函数，⽤于获取和读取Fashion-MNIST数据集。它返回训练集
# 和验证集的数据迭代器。此外，它还接受⼀个可选参数，⽤来将图像⼤小调整为另⼀种形状。
def load_data_fashion_mnist(batch_size,resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()] # 转换为tensor类型 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    if resize:
        trans.insert(0, transforms.Resize(resize)) # 如果有Resize参数传进来，就进行resize操作，如果resize有值，那么需要进行图片的resize
    trans = transforms.Compose(trans) # Compose操作就是多种操作串联组合起来
    mnist_train = torchvision.datasets.FashionMNIST(root="../01_Data/01_DataSet_FashionMNIST",train=True,transform=trans,download=True) # 加载数据，从上一级别目录中加载
    mnist_test = torchvision.datasets.FashionMNIST(root="../01_Data/01_DataSet_FashionMNIST",train=False,transform=trans,download=True) # 加载数据，从上一级别目录中加载
    train_iter = data.DataLoader(mnist_train,batch_size, shuffle=True, num_workers=get_dataloader_workers())
    test_iter = data.DataLoader(mnist_test,batch_size, shuffle=False, num_workers=get_dataloader_workers())
    return train_iter, test_iter # 返回两个迭代器

# 接下来，我们实现 softmax函数，并使���它来对 Fashion-MNIST数据集中的图像进行预测。
batch_size = 256
# ---------------第一步，训练集、测试集抽取----------------------
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# ---------------第二步，展平每个图像，将它们视为长度784的向量。向量的每个元素与w相乘，所以w也需要784行----------
#  因为数据集有10个类别，所以网络输出维度为10.
# 初始化w和b参数
num_inputs = 784
num_outputs = 10
w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

# ---------------第三步，softmax回归模型实现----------------------

def softmax(X):
    X_exp = torch.exp(X) # 每个都进行指数运算
    partition = X_exp.sum(1, keepdim=True) # 按行求和
    return X_exp / partition # 使用广播机制

# 使用softmax对 矩阵的数据进行归一化处理
# 将输入的图片数量整合为[任意行数,10]的矩阵
# -1为默认的批量大小，表示有多少个图片，每个图片用一维的784列个元素表示
def net(X):
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b) # -1为默认的批量大小，表示有多少个图片，每个图片用一维的784列个元素表示      

# 交叉熵损失的计算就是真实与预测之间的计算差值
# ----------------------第四步，交叉熵损失-----------------
def cross_entropy(y_hat, y):
    print(list(range(len(y_hat)))) # 将[0,1,...,len(y_hat)] 作为列表
    return -torch.log(y_hat[range(len(y_hat)),y]) # y_hat[range(len(y_hat)),y]为把y的标号列表对应的值拿出来。传入的y要是最大概率的标号

# ----------------第五步，准确率------------------
# 将预测类别与真实y元素进行比较
'''
这部分是测试案例
'''
y = torch.tensor([0,2]) # 标号索引
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]) # 两个样本在3个类别的预测概率   
# print("y_hat",y_hat[[0,1],y]) # 把第0个样本对应标号的预测值拿出来、第1个样本对应标号的预测值拿出来
# ------------------评估模型------------------
def accuracy(y_hat,y):
    """计算预测正确的数量"""
    # print("y:",y)
    # print("y_hat:",y_hat)
    # print("y_hat.shape:",y_hat.shape)
    # print("y_hat.shape[1]:",y_hat.shape[1])
    # print("len(y_hat.shape):",len(y_hat.shape)) # len返回的是行的数量
    # print("len(y_hat):",len(y_hat))# len返回的是行的数量
    # 经过这一步骤，
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # y_hat.shape[1]>1表示不止一个类别，每个类别有各自的概率
        y_hat = y_hat.argmax(axis=1) # 取每一行中概率最高的元素的下标,argmax就是取每个列中的最高概率的下标值
        print("y_hat:",y_hat)
    # print("y_hat.dtype:",y_hat.dtype)
    # print("y.dtype:",y.dtype)
    # print("y_hat.shape:",y_hat.type())
    # print("y.type:",y.type())
    # 这一步是在干嘛????
    cmp = y_hat.type(y.dtype) == y # 先判断逻辑运算符==，再赋值给cmp，cmp为布尔类型的数据
    print("cmp:",cmp)
    # 也就是y.dtype是int64类型，那么True转化为1，false转化为0，相加就是1
    return float(cmp.type(y.dtype).sum()) # 获得y.dtype的类型作为传入参数，将cmp的类型转为y的类型（int型），然后再求和

'''
这部分是测试案例
'''
# print("corss_entropy:",cross_entropy(y_hat,y))
# print("accuracy(y_hat,y) / len(y):",accuracy(y_hat,y) / len(y))
# print("accuracy(y_hat,y):",accuracy(y_hat,y))
# print("len(y):",len(y))


# 继续封装
# 可以评估在任意模型net的准确率# 可以评估在任意模型net的准确率
# ---------------------评估模型准确率------------------------
def evaluate_accuracy(net,data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module): # 如果net模型是torch.nn.Module实现的神经网络的话，将它变成评估模式     
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数，metric为累加器的实例化对象，里面存了两个数
    for X, y in data_iter:
        metric.add(accuracy(net(X),y),y.numel()) # net(X)将X输入模型，获得预测值。y.numel()为样本总数
    return metric[0] / metric[1] # 分类正确的样本数 / 总样本数


# Accumulator实例中创建了2个变量，用于分别存储正确预测的数量和预测的总数量
class Accumulator:
    """在n个变量上累加"""
    def __init__(self,n):
        self.data = [0,0] * n
        
    def add(self, *args):
        self.data = [a+float(b) for a,b in zip(self.data,args)] # zip函数把两个列表第一个位置元素打包、第二个位置元素打包....
        
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self,idx):
        return self.data[idx]
    


# print(evaluate_accuracy(net, test_iter))

# --------------------训练函数------------------------
def train_epoch_ch3(net,train_iter,loss,updater):
    if(isinstance(net,torch.nn.Module)):
        net.train()  # 将模型设置为训练模式
    metric = Accumulator(3) # ��加器，存了3个值：loss��加器、正确预测数��加器、样本总数��加器
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y) # 计算损失
        if isinstance(updater,torch.optim.Optimizer): # 如果updater是pytorch的优化器的话
            updater.zero_grad() # ���度清零
            l.backward() # 反向传播
            updater.step() # 进行一步��度下降
            metric.add(float(l) * len(y), accuracy(y_hat,y),y.size().numel()) # 总的训练损失、样本正确数、样本总数  
        else: # 如果updater是pytorch的SGD优化器的话
            l.sum().backward() # 反向传播
            updater(X.shape[0]) # 进行一步SGD下降
            metric.add(float(l.sum()), accuracy(y_hat,y), y.numel()) # 总的训练��失、样本正确数、样本总数
    return metric[0] / metric[2] , metric[1] / metric[2]  # 所有loss累加除以样本总数，总的正确个数除以样本总数  


# 动画绘制，就是绘制图片
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear',yscale='linear',
                fmts=('-','m--','g-.','r:'),nrows=1,ncols=1,
                figsize=(3.5,2.5)): 
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: d2l.set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)         
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)] 
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
# 总训练函数
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],       
                       legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
    train_loss, train_acc = train_metrics
    # test_loss, test_acc = test_acc
    # print(f'Train loss {train_loss:.3f}, train acc {train_acc:.3f}')
    # print(f'Test loss {test_loss:.3f}, test acc {test_acc:.3f}')


# 小批量随即梯度下降来优化模型的损失函数
lr = 0.1
# 用sgd更新w和b参数
def updater(batch_size):
    return d2l.sgd([w,b],lr,batch_size)

num_epochs = 10

# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# 预测数据
def predict_ch3(net,test_iter,n=6):
    for X, y in test_iter: 
        break # 仅拿出一批六个数据
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])
    d2l.plt.show()
    
predict_ch3(net,test_iter)
