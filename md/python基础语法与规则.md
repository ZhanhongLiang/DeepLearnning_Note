# Numpy常用方法

## permutation方法

```python
>>> import numpy as np
>>> m = 5
>>> permutation = list(np.random.permutation(m))
>>> permutation
[3, 2, 4, 0, 1] 
# new row 0 is old row 3
# new row 1 is old row 2
# new row 2 is old row 4
# new row 3 is old row 0
# new row 4 is old row 1
# m**2代表m^2 = 25
>>> X = np.arange(m**2).reshape((m,m)) 
>>> X
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
# 代表X的行按照permutation的下标进行重新排列
# [15, 16, 17, 18, 19]代表下标3
# [10, 11, 12, 13, 14]代表下标2
# [20, 21, 22, 23, 24]代表下标4
# [ 0,  1,  2,  3,  4]代表下标0
# [ 5,  6,  7,  8,  9]代表下标1
>>> X[permutation,:]
array([[15, 16, 17, 18, 19],
       [10, 11, 12, 13, 14],
       [20, 21, 22, 23, 24],
       [ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9]])
# 代表X的列按照permutation的下标进行重新排列
# [ 0,  1,  2,  3,  4]
#   ^   ^   ^   ^   ^
#   0   1   2   3   4
#         变成
# [ 3,  2,  4,  0,  1]
#   ^   ^   ^   ^   ^
#   3   2   4   0   1
# 
# [ 5,  6,  7,  8,  9]
#   ^   ^   ^   ^   ^
#   0   1   2   3   4
# 变成
# [ 8,  7,  9,  5,  6]
#   ^   ^   ^   ^   ^
#   3   2   4   0   1
>>> X[:,permutation]
array([[ 3,  2,  4,  0,  1],
       [ 8,  7,  9,  5,  6],
       [13, 12, 14, 10, 11],
       [18, 17, 19, 15, 16],
       [23, 22, 24, 20, 21]])
>>> X[:,permutation].reshape((1,m**2))
array([[ 3,  2,  4,  0,  1,  8,  7,  9,  5,  6, 13, 12, 14, 10, 11, 18,
        17, 19, 15, 16, 23, 22, 24, 20, 21]])```
```

## pad方法

```python
# 比如：
# 导入numpy库进行数值计算
import numpy as np
# 创建一个3维数组，包含3个2维子数组
arr3D = np.array([[[1, 1, 2, 2, 3, 4],
             [1, 1, 2, 2, 3, 4], 
             [1, 1, 2, 2, 3, 4]], 
             
            [[0, 1, 2, 3, 4, 5], 
             [0, 1, 2, 3, 4, 5], 
             [0, 1, 2, 3, 4, 5]], 
             
            [[1, 1, 2, 2, 3, 4], 
             [1, 1, 2, 2, 3, 4], 
             [1, 1, 2, 2, 3, 4]]])

# 打印边界填充后的结果
print('constant:  \n' + str(np.pad(arr3D, ((0, 0), (1, 1), (2, 2)), 'constant')))
```

> 参考链接: https://blog.csdn.net/zenghaitao0128/article/details/78713663n

## randn用法

```python
print("=====我们来测试一下=====")
np.random.seed(1)
x = np.random.randn(4,3,3,2)

print ("x.shape =", x.shape)
print("x = " , x)
print ("x[1, 1] =", x[1, 1])

# 绘制图
fig , axarr = plt.subplots(1,2)  # 一行两列
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
plt.show()
```

> 参考链接:https://juejin.cn/post/7115205069120733215

# 基础语法

## 1.1 yield与next的搭配使用

> 要理解yield的意思必须先第一步将yield看成是return的意思

## 1.2 矩阵基础语法

### 1.2.1 np.power用法

> np.power是矩阵的幂函数用法

## 1.3` Pyhton装饰器`

>由于函数也是一个对象，而且函数对象可以被赋值给变量，所以，通过变量也能调用该函数

[python 装饰器的作用](https://liaoxuefeng.com/books/python/functional/decorator/index.html)

#### `带参数的装饰器`

>`如果decorator本身需要传入参数,那就需要编写一个返回decorator的高阶函数,写出来会更复杂`

```python
def log(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator

@log('execute')
def now():
    print('2024-6-1')
now()
```

>我们来剖析上面的语句，首先执行`log('execute')`，返回的是`decorator`函数，再调用返回的函数，参数是`now`函数，返回值最终是`wrapper`函数。
>
>以上两种decorator的定义都没有问题，但还差最后一步。因为我们讲了函数也是对象，它有`__name__`等属性，但你去看经过decorator装饰之后的函数，它们的`__name__`已经从原来的`'now'`变成了`'wrapper'`：

```
>>> now.__name__
'wrapper'
```

>因为返回的那个`wrapper()`函数名字就是`'wrapper'`，所以，需要把原始函数的`__name__`等属性复制到`wrapper()`函数中，否则，有些依赖函数签名的代码执行就会出错。

#### 案例判断

- 判断以下代码,这个代码是BERT里面LoadingSingleClassion.py摘抄的

```python
    
   # 这个是python装饰器
   # 在 data_process上面定义了@process_cache(unique_key=["max_sen_len"])
   # 定义了一个带process_cache带参数的装饰器函数
def process_cache(unique_key=None):
    """
    数据预处理结果缓存修饰器
    :param : unique_key
    :return:
    """
    if unique_key is None:
        raise ValueError(
            "unique_key 不能为空, 请指定相关数据集构造类的成员变量，如['top_k', 'cut_words', 'max_sen_len']")
    def decorating_function(func):
        def wrapper(*args, **kwargs):
            logging.info(f" ## 索引预处理缓存文件的参数为：{unique_key}")
            obj = args[0]  # 获取类对象，因为data_process(self, file_path=None)中的第1个参数为self
            file_path = kwargs['file_path'] # 相当于多参数,获取名字为file_paht的参数
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
            paras = f"cache_{file_name}_"
            for k in unique_key:
                paras += f"{k}{obj.__dict__[k]}_"  # 遍历对象中的所有参数
            cache_path = os.path.join(file_dir, paras[:-1] + '.pt')
            start_time = time.time()
            # 如果不存在.pt文件，也就是处理的文件模型
            # 这里就是装饰器强大之处,如果不存在cache_train_max_sen_lenNone.pt文件
            # 需要执行data=func(*args,**kwargs),生成这个模型
            if not os.path.exists(cache_path):
                logging.info(f"缓存文件 {cache_path} 不存在，重新处理并缓存！")
                data = func(*args, **kwargs)
                # 这里是'wb'是write bin的意思,就是以二进制形式写入
                with open(cache_path, 'wb') as f:
                    torch.save(data, f)
            else:
                logging.info(f"缓存文件 {cache_path} 存在，直接载入缓存文件！")
                with open(cache_path, 'rb') as f:
                    data = torch.load(f)
            end_time = time.time()
            logging.info(f"数据预处理一共耗时{(end_time - start_time):.3f}s")
            return data

        return wrapper

    return decorating_function
    
    
    """
    tqdm库，用于显示python库训练的进度
    只是一个三方库
    """
    # data_process()相当于调用了
    # data_process = process_cache(unique_key=["max_sen_len"])(data_process)
    # 相当于首先执行了process_cache(unique_key=["max_sen_len"]),返回的是decorating_function函数
    # 再调用返回的函数,参数是data_process(self, file_path=None),返回值是wrapper函数
    # 也就是最后data_process指向了wrapper函数
    @process_cache(unique_key=["max_sen_len"])
    def data_process(self, file_path=None):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param file_path: 数据集路径
        :return:
        """
        # open函数
        # 且调用readlines()函数
        # 调用readline()可以每次读取一行内容，调用readlines()一次读取所有内容并按行返回list。因此，要根据需要决定怎么调用。
        raw_iter = open(file_path, encoding="utf8").readlines()
        data = []
        max_len = 0
        # tqdm库，就是按照
        for raw in tqdm(raw_iter, ncols=80):
            # 取得文本和标签
            line = raw.rstrip("\n").split(self.split_sep)
            s, l = line[0], line[1]
            # python的迭代器,在最前面加上分类标志位[CLS]
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
            # 用来对Token序列进行截取，最长为max_position_embeddings个字符,默认为512
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            # 在末尾处添加上[SEP]符号
            tmp += [self.SEP_IDX]
            # 将tmp转化为张量
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            # 将标签转化为张量
            l = torch.tensor(int(l), dtype=torch.long)
            # 保存最长序列的长度
            max_len = max(max_len, tensor_.size(0))
            # 将文本张量和标签张量添加到data里面
            data.append((tensor_, l))
        return data, max_len
```

>参照这篇文章 [Python装饰器执行顺序](https://kingname.info/2023/04/16/order-of-decorator/)
>
>`执行顺序:process(unique_key=["max_sen_len"])(data_process)`
>
>看代码逻辑,
>
>```python
>            # 如果不存在.pt文件，也就是处理的文件模型
>            # 这里就是装饰器强大之处,如果不存在cache_train_max_sen_lenNone.pt文件
>            # 需要执行data=func(*args,**kwargs),生成这个模型
>            if not os.path.exists(cache_path):
>                logging.info(f"缓存文件 {cache_path} 不存在，重新处理并缓存！")
>                data = func(*args, **kwargs)
>                # 这里是'wb'是write bin的意思,就是以二进制形式写入
>                with open(cache_path, 'wb') as f:
>                    torch.save(data, f)
>            else
>```
>
>`这里的func(*args,**kwargs)就是调用了data_process函数`
>
>`这里的*args就是self,**kwargs就是多参数,file_path = kwargs['file_path'] # 相当于多参数,获取名字为file_paht的参数，这句话的意思就是获取kwargs['files_path']里面是'files_path'名字的值`,
>最后torch.save(data,f),`data是data_process返回的参数赋予给data`
>
>然后以二进制的形式写成.pt格式



# Pytorch+anaconda 配置

## 1.1 激活环境

```
conda create -n 环境名 --clone base
```

## 1.2 进入环境

```
conda activate 环境名
```



## 1.3 设置默认环境(选做)

>可以修改/.bashrc文件的配置，即在/.bashrc文件的末尾添加如下一行：

```
source activate 环境名
```

>保存后关闭当前窗口，并重新打开新的窗口，可以看到默认启动环境已经不再是base了：



## 1.4 安装Pytorch(待续)

>

# Pytorch基础语法

## 1.torchvision.transforms

### 1.1 transforms.ToTensor()

> 把一个取值范围是`[0,255]`的`PIL.Image`或者`shape`为`(H,W,C)`的`numpy.ndarray`，转换成形状为`[C,H,W]`，取值范围是`[0,1.0]`的`torch.FloadTensor`

```python
data = np.random.randint(0, 255, size=300)
img = data.reshape(10,10,3)
print(img.shape)
img_tensor = transforms.ToTensor()(img) # 转换成tensor
print(img_tensor)
```

### 1.2 transforms.Resize(x)

> 简单来说就是**调整PILImage对象的尺寸**，注意不能是用io.imread或者cv2.imread读取的图片，这两种方法得到的是ndarray。
>
> 将图片短边缩放至x，长宽比保持不变

```python
transforms.Resize(x)
```

> 而一般输入深度网络的特征图长宽是相等的，就不能采取等比例缩放的方式了，需要同时指定长宽：

```python
transforms.Resize([h, w])
```

> 例如transforms.Resize([224, 224])就能将输入图片转化成224×224的输入特征图。
>
> 这样虽然会改变图片的长宽比，但是本身并没有发生裁切，仍可以通过resize方法返回原来的形状：

```python
from PIL import Image
from torchvision import transforms

img = Image.open('1.jpg')
w, h = img.size
resize = transforms.Resize([224,244])
img = resize(img)
img.save('2.jpg')
resize2 = transforms.Resize([h, w])
img = resize2(img)
img.save('3.jpg')
```

### 1.3 transforms.Compose(transforms)

> 将多个`transform`组合起来使用。
>
> `transforms`： 由`transform`构成的列表. 例子：
>
> 
>
> 本文的主题是其中的torchvision.transforms.Compose()类。这个类的主要作用是串联多个图片变换的操作。
>
> 这个类的构造很简单：



```python
class torchvision.transforms.Compose(transforms):
 # Composes several transforms together.
 # Parameters: transforms (list of Transform objects) – list of transforms to compose.
 
Example # 可以看出Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作。
>>> transforms.Compose([
>>>     transforms.CenterCrop(10),
>>>     transforms.ToTensor(),])
```

> 事实上，Compose()类会将transforms列表里面的transform操作进行遍历。实现的代码很简单：

```python
## 这里对源码进行了部分截取。
def __call__(self, img):
    for t in self.transforms:   
        img = t(img)
    return img
```

## 2. torch.utils.data

### 2.1 torch.utils.data.DataLoader()

> 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
```

**参数：**

- **dataset** (*Dataset*) – 加载数据的数据集。
- **batch_size** (*int*, optional) – 每个batch加载多少个样本(默认: 1)。
- **shuffle** (*bool*, optional) – 设置为`True`时会在每个epoch重新打乱数据(默认: False).
- **sampler** (*Sampler*, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略`shuffle`参数。
- **num_workers** (*int*, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
- **collate_fn** (*callable*, optional) –
- **pin_memory** (*bool*, optional) –
- **drop_last** (*bool*, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

> https://blog.csdn.net/zfhsfdhdfajhsr/article/details/116836851

## 3.torch.tensor

### 3.1 matmul

> orch.matmul是tensor的乘法，输入可以是高维的。
> 当输入都是二维时，就是普通的[矩阵乘法](https://so.csdn.net/so/search?q=矩阵乘法&spm=1001.2101.3001.7020)，和tensor.mm函数用法相同。

```python
import torch
a = torch.ones(3,4)
b = torch.ones(4,2)
c = torch.matmul(a,b)
print(c.shape)
```

```
size([3,2])
```

### 3.2 `view`

>`view`用法很重要，必须掌握这个用法
>
>参考这篇文章
>
>[view()用法](https://www.cnblogs.com/zhangxuegold/p/17504649.html)

### 函数简介

>Pytorch中的view函数主要用于Tensor维度的重构，即返回一个有相同数据但不同维度的Tensor。
>
>根据上面的描述可知，view函数的操作对象应该是Tensor类型。如果不是Tensor类型，可以通过`tensor = torch.tensor(data)`来转换。

#### 普通用法 (手动调整size)

>`view(参数a,参数b,…)`，其中，总的参数个数表示将张量重构后的维度。
>
>`view()`相当于`reshape`、`resize`，重新调整Tensor的形状。

```python
import torch
a1 = torch.arange(0,16)
print(a1) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

a2 = a1.view(8, 2) # 将a1的维度改为8*2
a3 = a1.view(2, 8) # 将a1的维度改为2*8
a4 = a1.view(4, 4) # 将a1的维度改为4*4

# a5 = a1.view(2,2,1,4)
# 更多的维度也没有问题，只要保证维度改变前后的元素个数相同就行,即 2*2*1*4=16。

print(a2)
print(a3)
print(a4)
```

```
tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15]])
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]])
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
```

#### 特殊用法(自动调节size)

>`view(参数a,参数b,…)`中一个参数定为-1，代表自动调整这个维度上的元素个数，则表示该维度取决于其它维度，由Pytorch自己补充，以保证元素的总数不变。

```python
import torch
a1 = torch.arange(0,16)
print(a1) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
a2 = a1.view(-1, 16)
a3 = a1.view(-1, 8)
a4 = a1.view(-1, 4)
a5 = a1.view(-1, 2)
a6 = a1.view(4*4, -1)
a7 = a1.view(1*4, -1)
a8 = a1.view(2*4, -1)

print(a2)
print(a3)
print(a4)
print(a5)
print(a6)
print(a7)
print(a8)
```

```
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]])
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]])
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15]])
tensor([[ 0],
        [ 1],
        [ 2],
        [ 3],
        [ 4],
        [ 5],
        [ 6],
        [ 7],
        [ 8],
        [ 9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15]])
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15]])
```

>`view(-1)`表示将Tensor转为一维Tensor。

```python
a9 = a1.view(-1)

print(a1)
print(a9) # 因此，转变后还是一维，没什么变换
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
```

到此这篇关于pytorch中的 .view()函数的用法介绍的文章就介绍到这了,更多相关pytorch .view()函数内容请去pytorch官网文档查看。

### 3.3 `masked_fill()函数`

**1. 函数形式**

```python
torch.Tensor.masked_fill(mask, value)
```

>**2. 函数功能**
>输入的mask*m**a**s**k*需要与当前的基础Tensor的形状一致。
>将mask*m**a**s**k*中为True的元素对应的基础Tensor的元素设置为值value*v**a**l**u**e*。
>
>**3. 函数参数**
>
>- **mask**：mask既可以是int型Tensor（值为0或者1）也可以是bool型Tensor（值为False或者True）
>- **value**：float，填充的值
>
>**4. 函数返回值**
>返回填充后的Tensor

>下面一个简单的例子说明masked_fill函数的使用，首先我们创建一个4x4的一个基础矩阵，然后创建一个4x4的对角矩阵，然后根据对角矩阵将对角线上的基础机矩阵的值全部设置为100，具体的代码如下所示。

```python
import torch

if __name__ == '__main__':
    tensor = torch.arange(0,16).view(4,4)
    print('origin tensor:\n{}\n'.format(tensor))

    mask = torch.eye(4,dtype=torch.bool)
    print('mask tensor:\n{}\n'.format(mask))

    tensor = tensor.masked_fill(mask,100)
    print('filled tensor:\n{}'.format(tensor))

```

```
origin tensor:
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])

mask tensor:
tensor([[ True, False, False, False],
        [False,  True, False, False],
        [False, False,  True, False],
        [False, False, False,  True]])

filled tensor:
tensor([[100,   1,   2,   3],
        [  4, 100,   6,   7],
        [  8,   9, 100,  11],
        [ 12,  13,  14, 100]])

```

### 3.4  `cat()`

>一般`torch.cat()`是为了把多个`tensor`进行拼接而存在的。实际使用中，和`torch.stack()`使用场景不同：参考链接[torch.stack()](https://blog.csdn.net/xinjieyuan/article/details/105205326)，但是本文主要说`cat()`。
>
>`torch.cat()` 和`python`中的内置函数`cat()`， 在使用和目的上，是没有区别的，区别在于前者操作对象是`tensor`。
>
>cat()
>
>函数目的： 在给定维度上对输入的张量序列seq 进行连接操作。
>
>outputs = torch.cat(inputs, dim=?) → Tensor
>  1.参数
>
>​         inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
>​         dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。
>
>2. 重点
>输入数据必须是序列，序列中数据是任意相同的shape的同类型tensor
>维度不可以超过输入数据的任一个张量的维度

```python
import torch

x1 = torch.tensor([[11,21,31],[21,31,41]],dtype=torch.int)
# x1.shape # torch.Size([2, 3])
# x2
x2 = torch.tensor([[12,22,32],[22,32,42]],dtype=torch.int)

# x2.shape  # torch.Size([2, 3])

inputs = [x1,x2]
print(inputs)

output = torch.cat(inputs,dim=0)
print(output)
print(output.shape) # shape [4,3]

output = torch.cat(inputs,dim=1)
print(output)
print(output.shape) # shape [2,6]

# output = torch.cat(inputs,dim=2)

```

```
[tensor([[11, 21, 31],
        [21, 31, 41]], dtype=torch.int32), tensor([[12, 22, 32],
        [22, 32, 42]], dtype=torch.int32)]
tensor([[11, 21, 31],
        [21, 31, 41],
        [12, 22, 32],
        [22, 32, 42]], dtype=torch.int32)
torch.Size([4, 3])
tensor([[11, 21, 31, 12, 22, 32],
        [21, 31, 41, 22, 32, 42]], dtype=torch.int32)
torch.Size([2, 6])
```

## 4.torch.nn

### init

> 随机初始化权重参数算法

当然可以，以下是每种初始化方法的用途、代码实现、数学公式以及进一步的解释：

#### 1. **均匀分布初始化 (`torch.nn.init.uniform_`)**
   - **用途**：用于从均匀分布中随机初始化权重。适用于在不确定最佳初始化范围时，为所有权重赋予一个均匀分布的值。

   - **公式**：
     $$
     (\mathbf{W}_{ij} \sim \mathcal{U}(a, b))
     $$
     
     
     - 
     
       
       $$
       (\mathcal{U}(a, b))
       $$
       表示均匀分布，区间为 \([a, b]\)。
     
   - **代码示例**：
     
     ```python
     import torch
import torch.nn as nn
     
     tensor = torch.empty(3, 5)
     nn.init.uniform_(tensor, a=0, b=1)
     ```
     
   - **解释**：将张量 `tensor` 的每个元素初始化为范围在 \([0, 1]\) 的均匀随机值。这种方法的简单性使其适用于各种场景。

#### 2. **正态分布初始化 (`torch.nn.init.normal_`)**
   - **用途**：用于从正态分布中随机初始化权重，适合期望权重值围绕某个均值对称分布的情况。

   - **公式**：
     $$
     (\mathbf{W}_{ij} \sim \mathcal{N}(\mu, \sigma^2))
     $$
     
     
     - 
       $$
       (\mathcal{N}(\mu, \sigma^2))
       $$
        表示均值为 \(\mu\)、方差为 \(\sigma^2\) 的正态分布。
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 5)
     nn.init.normal_(tensor, mean=0, std=1)
     ```
     
   - **解释**：将张量 `tensor` 的每个元素初始化为均值为 0，标准差为 1 的正态分布随机值。这种初始化在实践中经常使用，尤其在期望值接近零的情况下效果良好。

#### 3. **常量初始化 (`torch.nn.init.constant_`)**
   - **用途**：将所有权重初始化为相同的常量值，通常用于偏置初始化。

   - **公式**：
     $$
     (\mathbf{W}_{ij} = c)
     $$
     
     
     - \(c\) 是一个常数值。
     
   - **代码示例**：
     ```python
     tensor = torch.empty(3, 5)
     nn.init.constant_(tensor, 0.1)
     ```
     
   - **解释**：将张量 `tensor` 的每个元素初始化为 0.1。常量初始化通常用于网络中的偏置，因为偏置的初始值可能不需要变化太大。

#### 4. **单位矩阵初始化 (`torch.nn.init.eye_`)**
   - **用途**：初始化二维张量为单位矩阵，通常用于某些线性层的特殊初始化场景。

   - **公式**：
     $$
     (\mathbf{W} = \mathbf{I}_n)
     $$
     
     
     - 
       $$
       (\mathbf{I}_n)
       $$
        是 \(n\) 维单位矩阵。
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 3)
     nn.init.eye_(tensor)
     ```
     
   - **解释**：将 `tensor` 初始化为一个 \(3 \times 3\) 的单位矩阵（对角线为 1，其他元素为 0）。这种初始化通常在某些特定的线性变换或 RNN 层中有用。

5. **Xavier 均匀分布初始化 (`torch.nn.init.xavier_uniform_`)**

   - **用途**：用于避免梯度消失或爆炸，特别是在深层网络中。此方法考虑了输入和输出的规模。

   - **公式**：
     $$
     (\mathbf{W}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right))
     $$
     
     
     - 
       $$
       (n_{in}) 和 (n_{out})
       $$
        分别为输入和输出的单元数。
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 5)
     nn.init.xavier_uniform_(tensor)
     ```
     
   - **解释**：将权重初始化为在 \(\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]\) 之间的均匀分布随机值。这种初始化方式通常适用于带有 Sigmoid 或 Tanh 激活函数的神经网络。

#### 6. **Xavier 正态分布初始化 (`torch.nn.init.xavier_normal_`)**
   - **用途**：与 Xavier 均匀分布类似，但使用正态分布进行初始化，适用于深度神经网络。

   - **公式**：
     $$
     (\mathbf{W}_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right))
     $$
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 5)
     nn.init.xavier_normal_(tensor)
     ```
     
   - **解释**：将权重初始化为均值为 0，方差为
     $$
     (\frac{2}{n_{in} + n_{out}})
     $$
      的正态分布随机值。这种方法同样适用于具有 Sigmoid 或 Tanh 激活函数的神经网络。

#### 7. **Kaiming 均匀分布初始化 (`torch.nn.init.kaiming_uniform_`)**
   - **用途**：适用于 ReLU 及其变体激活函数的网络初始化，考虑了非线性激活的影响。

   - **公式**：
     $$
     (\mathbf{W}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right))
     $$
     
     
     - 这里的 
       $$
       (n_{in})
       $$
        是输入单元数。
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 5)
     nn.init.kaiming_uniform_(tensor, nonlinearity='relu')
     ```
     
   - **解释**：将权重初始化为在 \(\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]\) 之间的均匀分布随机值。Kaiming 初始化通过考虑 ReLU 激活函数的特性，确保了在深层网络中更稳定的训练。

#### 8. **Kaiming 正态分布初始化 (`torch.nn.init.kaiming_normal_`)**
   - **用途**：与 Kaiming 均匀分布初始化类似，但使用正态分布进行初始化，适用于使用 ReLU 激活函数的深度神经网络。

   - **公式**：
     $$
     (\mathbf{W}_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right))
     $$
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 5)
     nn.init.kaiming_normal_(tensor, nonlinearity='relu')
     ```
     
   - **解释**：将权重初始化为均值为 0，方差为 \(\frac{2}{n_{in}}\) 的正态分布随机值。这种方法尤其适合于深层网络，特别是使用 ReLU 及其变体激活函数的网络。

#### 9. **Orthogonal 初始化 (`torch.nn.init.orthogonal_`)**
   - **用途**：确保权重矩阵的正交性，这在某些情况下可以提高训练的稳定性和模型性能。

   - **公式**：
     $$
     (\mathbf{W}\mathbf{W}^T = \mathbf{I})
     $$
     
   - **代码示例**：
     
     ```python
     tensor = torch.empty(3, 5)
     nn.init.orthogonal_(tensor)
     ```
     
   - **解释**：将张量 `tensor` 初始化为正交矩阵，使其转置矩阵与自身相乘后为单位矩阵。这种初始化在某些特殊网络结构中表现良好，如 RNN 和自编码器。

#### 10. **Sparse 初始化 (`torch.nn.init.sparse_`)**
   - **用途**：初始化稀疏矩阵，使得权重矩阵中大部分元素为零，这对于某些需要稀疏表示的网络结构有用。

   - **公式**：
     $$
     (\mathbf{W}_{ij} = \begin{cases}
      \mathcal{N}(0, \sigma^2) & \text{以概率 \(p\) 保留非零元素} \\
      0 & \text{以概率 \(1-p\) 置为零}
      \end{cases})
$$
   
   
   - **代码示例**：
     ```python
     tensor = torch.empty(10, 10)
     nn.init
     ```

### 5. MSELoss

`MSELoss` 是 PyTorch 中用于计算均方误差 (Mean Squared Error, MSE) 的损失函数。MSE 是神经网络训练中常用的一种回归损失函数，它计算预测值和目标值之间的差异的平方和的平均值。其公式如下：

### 数学公式
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
-
$$
  y_i 是第 i 个目标值（真实值）。
$$
- 
$$
   \hat{y}_i  是第 i 个预测值。
  $$


- n  是样本的总数。

- 代码示例

```python
import torch
import torch.nn as nn

# 定义 MSELoss 函数
criterion = nn.MSELoss()

# 创建一些示例数据
predicted = torch.tensor([2.5, 0.0, 2.0, 8.0])  # 模型的预测值
target = torch.tensor([3.0, -0.5, 2.0, 7.0])    # 真实目标值

# 计算 MSE Loss
loss = criterion(predicted, target)
print(loss)
```

- 解释

- **用途**：`MSELoss` 通常用于回归任务中，在这种任务中，模型预测的是连续值而非类别。MSE 计算预测值和真实值之间的误差并对其平方，因此更关注大的误差，因为平方会放大它们的影响。
- **优势**：简单且易于解释，损失值越小，预测值与目标值越接近。
- **缺点**：由于平方放大了误差，MSE 对于异常值（outliers）非常敏感，这可能会导致模型偏向那些异常值。

`MSELoss` 是构建回归模型时的重要组成部分，通常与优化器（如 SGD 或 Adam）一起使用，以最小化模型在训练集上的误差。

### linear

>参考这篇文章[linear用法及原理](https://blog.csdn.net/zhaohongfei_358/article/details/122797190)