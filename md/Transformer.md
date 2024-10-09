# 1. Transformer

> Transformer 是一个基于自注意力的序列到序列模型，与基于循环神经网络的序列到序列模型不同，其可以能够并行计算。



## 1.1 序列到序列模型

> 1. 输入和输出长度一样
> 2. 机器决定输出的长度

> 应用场景:
>
> 1. 语音识别：`输入`是声音信号，`输出`是语音识别的结果，即`输入的这段声音信号所对应的文字`。
> 2. 机器翻译：机器`输入`一个语言的句子，`输出`另外一个语言的句子，输入句子的长度是N，输出句子的长度是N', N 跟 N′之间的关系由机器决定。
> 3. 语音翻译：我们对机器说一句话，比如“machine learning”，机器直接把听到的英语的声
>    音信号翻译成中文。
> 4. 语音合成: 输入文字、输出声音信号就是语音合成（Text-To-Speech，TTS）。现在还没有真的做端
>    到端（end-to-end）的模型，以闽南语的语音合成为例，其使用的模型还是分成两阶，首先模
>    型会先把白话文的文字转成闽南语的拼音，再把闽南语的拼音转成声音信号。从闽南语的拼
>    音转成声音信号这一段是通过序列到序列模型 echotron 实现的。
> 5. 聊天机器人：![](https://pic.superbed.cc/item/66d85610fcada11d375a3b39.png)
> 6. 问答任务:
>
> ![](https://pic.superbed.cc/item/66d85699fcada11d375a4058.png)
>
> 7. 句法分析
>
> 在句法分析的任务中，输入是一段文字，输出是一个树状的结构，而一个树状的结构可以看成一个序列，该序列代表了这个树的结构把树的结构转成一个序列以后，我们就可以用序列到序列模型来做句法分析，具体可参考论文“Grammar as a Foreign Language”
>
> ![](https://pic.superbed.cc/item/66d8583bfcada11d375a489e.png)
>
> 8. 多标签分类
>
> ![](https://pic.superbed.cc/item/66d858c0fcada11d375a4b6a.png)
>
> ![](https://pic.superbed.cc/item/66d85a57fcada11d375a5fb7.png)

## 1.2 Transformer 结构

![img](file:///C:/Users/25212/OneDrive/Transformer结构图.png)

> 整个Transformer结构如这个结构化所示

### 1.2.1 Embedding layer (输入嵌入)

> 假设输入一句"Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"
>
> 通俗的理解:输入是序列只是一段句子，我们是需要将他转化为张量才能输入到模型中,那么就需要进行词的分割，且转化为张量
>
> 精准的定义: Embedding layer可以将看成将序列数据从`高维空间`转化为`低维空间`，或者理解为`高维的稀疏矩阵`转化为`低维的稠密矩阵`

> Embedding layer的方法:
>
> - `ONE-HOT`
> - PCA降维
> - `Word2Vec`
> - Item2Vec
> - 矩阵分解
> - 基于深度学习的协同过滤

> Let us create the embedding from scratch

#### 1. Vocabulary ONE-HOT 2 lower dim Vector

> `我们需要先建立vocabulary 词典的映射关系，因为embedding layer是需要建立词典的映射关系`,然后后续的句子是通过词典里面来寻找的
>
> 因为句子是一个一个单词组成的，经典的思想就是将"句子"转化为词数组，这个时候就需要使用到我们经典的ONE-HOT编码

![img](file:///C:/Users/25212/OneDrive/Trans-One-hot编码.png)

> 根据上图，
>
> 1. 先Tokenize,那么Tokenize的代码如下

```python
'''
Author: Jean_Leung
Date: 2024-09-06 10:10:58
LastEditors: Jean_Leung
LastEditTime: 2024-09-06 11:21:21
FilePath: \LinearModel\transformer_pratice\positional_embedding.py
Description: 搞懂Transformer的Positional_embedding layer的运作方式

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''

# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# visualization packages
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

# Tokenize the input text into individual words
# 第一步先tokenize字符串，经典的就是按照每个词进行tokenize
# tokenize 就是将文本分割成离散的单元
def tokenize(sequence):
    # remove punctuation
    # 移除标点符号
    for punc in["!", ".", "?"]:
        sequence = sequence.replace(punc, "") # 用无字符取代标点位置
    # 返回一个字符数组
    return [token.lower() for token in sequence.split(" ")]

print(tokenize(example))
```

输出结果:

```
['hello', 'this', 'is', 'an', 'example', 'of', 'a', 'paragraph', 'that', 'has', 'been', 'split', 'into', 'its', 'basic', 'components', 'i', 'wonder', 'what', 'will', 'come', 'next', 'any', 'guesses']
```

> 2. 先用ONE-HOT建立词库映射，world:index 的关系

```python
# 下一步按照词的开头ASSIC进行排序，从a-z进行排序

def build_vocab(data):
    # 构建一个单词到index的映射
    vocab = list(set(tokenize(data)))
    # 进行排序，默认快排，从小到大排序
    # vocab.sort(key=lambda x: (ord(x[0]), x))
    vocab.sort()
    # # 建立一个index到单词的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

stoi = build_vocab(example) # 建立词库映射
print(stoi)
```

输出结果:

```
{'a': 0, 'an': 1, 'any': 2, 'basic': 3, 'been': 4, 'come': 5, 'components': 6, 'example': 7, 'guesses': 8, 'has': 9, 'hello': 10, 'i': 11, 'into': 12, 'is': 13, 'its': 14, 'next': 15, 'of': 16, 'paragraph': 17, 'split': 18, 'that': 19, 'this': 20, 'what': 21, 'will': 22, 'wonder': 23}
```

> 3. 现在已经根据映射建立了key:value的mapping关系，也就是one-hot编码，这个是词库的映射，接下我们需要生成`低维度的稠密矩阵`那么应该怎么生成呢?
>
> 实现代码可以参考这篇文章生成embedding matrix [word2vec的原理及实现（附github代码）](https://blog.csdn.net/qq_30189255/article/details/103049569)
>
> 实现简洁的基础版本代码参考这篇文章:[The Embedding Layer](https://medium.com/@hunter-j-phillips/the-embedding-layer-27d9c980d124)
>
> 原理可以参考这篇文章 [Word Embedding教程](https://fancyerii.github.io/books/word-embedding/)
>
> `这是演示过程，那么我们可以默认已经随机生成了E矩阵，然后通过`
>
> `这个过程就是高维稀疏矩阵通过矩阵乘法映射成低维稠密矩阵`
>
> ![img](file:///C:/Users/25212/OneDrive/E矩阵和onehot矩阵匹配.png)
>
> ![img](file:///C:/Users/25212/OneDrive/Embedding矩阵与onehot矩阵相乘.png)

```python
sequence = [stoi[word] for word in tokenize("I wonder what will come next!")]
print(sequence)

# Positional Encoding layer
# vocab size
# 获得 vocab_size
vocab_size = len(stoi)
# d_model 隐藏单元的个数
d_model = 3 # 数据是三维,这个时候embedding矩阵是(24,3)的

# 构建一个position的embedding
embedding = torch.rand(vocab_size,d_model) # size为(24,3)矩阵，因为需要进行运算
print(embedding)
embedded_sequence = embedding[sequence]
print("embedded_sequence:",embedded_sequence)
print("embedded_sequence.shape:",embedded_sequence.shape) # size为(6,3)
```

```
embedded_sequence: tensor([[0.6384, 0.0727, 0.9703],
        [0.3984, 0.1473, 0.1994],
        [0.8784, 0.3126, 0.1634],
        [0.5357, 0.7002, 0.5487],
        [0.0914, 0.3244, 0.3288],
        [0.1613, 0.3047, 0.1960]])
embedded_sequence.shape: torch.Size([6, 3])
```

![img](file:///C:/Users/25212/OneDrive/Embedding矩阵可视化.png)

> 4. 利用Pytorch自带的embedding进行生成矩阵
>
> ![img](file:///C:/Users/25212/OneDrive/Embedding过程.png)

```python
sequence = [stoi[word] for word in tokenize("I wonder what will come next!")]
# print(sequence)

# Positional Encoding layer
# vocab size
# 获得 vocab_size
vocab_size = len(stoi)
# d_model 隐藏单元的个数
d_model = 3 # 数据是三维

# 构建一个position的embedding
# embedding = torch.rand(vocab_size,d_model) # size为(24,3)矩阵，因为需要进行运算
lut  = nn.Embedding(vocab_size,d_model) # nn自带的Embedding类
lut.state_dict()['weight'] 
indices = torch.Tensor(sequence).long()
embedding = lut(indices)
print(embedding)
```

- 总代码

```python
'''
Author: Jean_Leung
Date: 2024-09-06 10:10:58
LastEditors: Jean_Leung
LastEditTime: 2024-09-06 14:35:22
FilePath: \LinearModel\transformer_practise\_token_embedding.py
Description: 搞懂Transformer的Positional_embedding layer的运作方式
              1. 从零开始建立词库映射
                1.1 先有单词库，然后我们利用One-hot映射成一个稀疏的高维矩阵
                1.2 再用一个转化矩阵将稀疏的高维矩阵转化为稠密的低维矩阵

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''


# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# visualization packages
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

# Tokenize the input text into individual words
# 第一步先tokenize字符串，经典的就是按照每个词进行tokenize
# tokenize 就是将文本分割成离散的单元
def tokenize(sequence):
    # remove punctuation
    # 移除标点符号
    for punc in["!", ".", "?"]:
        sequence = sequence.replace(punc, "")
    # 返回一个字符数组
    return [token.lower() for token in sequence.split(" ")]

# print(tokenize(example))

# 下一步按照词的开头ASSIC进行排序，从a-z进行排序

def build_vocab(data):
    # 构建一个单词到index的映射
    vocab = list(set(tokenize(data)))
    # 进行排序，默认快排，从小到大排序
    # vocab.sort(key=lambda x: (ord(x[0]), x))
    vocab.sort()
    # # 建立一个index到单词的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

stoi = build_vocab(example) # 建立词库映射
# print(stoi)

sequence = [stoi[word] for word in tokenize("I wonder what will come next!")]
# print(sequence)

# Positional Encoding layer
# vocab size
# 获得 vocab_size
vocab_size = len(stoi)
# d_model 隐藏单元的个数
d_model = 3 # 数据是三维

# 构建一个position的embedding
# embedding = torch.rand(vocab_size,d_model) # size为(24,3)矩阵，因为需要进行运算
lut  = nn.Embedding(vocab_size,d_model)
lut.state_dict()['weight'] # 返回字典dict
indices = torch.Tensor(sequence).long()
embedding = lut(indices)
print(embedding)
# embedded_sequence = embedding[sequence]
# print("embedded_sequence:",embedded_sequence)
# print("embedded_sequence.shape:",embedded_sequence.shape) # size为(6,3)
# 3D visualization of the positional encoding
# x,y,z = embedded_sequence[:,0],embedded_sequence[:,1],embedded_sequence[:,2]
# ax = plt.axes(projection='3d')
# ax.scatter3D(x,y,z)
# plt.show()
```

> 假设输入三段序列
>
> sequences = ["I wonder what will come next!",
>         "This is a basic example paragraph.",
>         "Hello, what is a basic split?"]
>
> ![img](file:///C:/Users/25212/OneDrive/Embedding过程.png)

```python
'''
Author: Jean_Leung
Date: 2024-09-06 10:10:58
LastEditors: Jean_Leung
LastEditTime: 2024-09-06 14:50:31
FilePath: \LinearModel\transformer_practise\_token_embedding.py
Description: 搞懂Transformer的Positional_embedding layer的运作方式
              1. 从零开始建立词库映射
                1.1 先有单词库，然后我们利用One-hot映射成一个稀疏的高维矩阵
                1.2 再用一个转化矩阵将稀疏的高维矩阵转化为稠密的低维矩阵

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''


# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# visualization packages
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

# Tokenize the input text into individual words
# 第一步先tokenize字符串，经典的就是按照每个词进行tokenize
# tokenize 就是将文本分割成离散的单元
def tokenize(sequence):
    # remove punctuation
    # 移除标点符号
    for punc in["!", ".", "?",","]:
        sequence = sequence.replace(punc, "")
    # 返回一个字符数组
    return [token.lower() for token in sequence.split(" ")]

# print(tokenize(example))

# 下一步按照词的开头ASSIC进行排序，从a-z进行排序

def build_vocab(data):
    # 构建一个单词到index的映射
    vocab = list(set(tokenize(data)))
    # 进行排序，默认快排，从小到大排序
    # vocab.sort(key=lambda x: (ord(x[0]), x))
    vocab.sort()
    # # 建立一个index到单词的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

stoi = build_vocab(example) # 建立词库映射
# print(stoi)

# sequence = [stoi[word] for word in tokenize("I wonder what will come next!")]
sequences = ["I wonder what will come next!",
             "This is a basic example paragraph.",
             "Hello, what is a basic split?"]

# tokenize the sequences
tokenized_sequences = [tokenize(seq) for seq in sequences]
print('tokenized sequences', tokenized_sequences)
# index the sequences 
indexed_sequences = [[stoi[word] for word in seq] for seq in tokenized_sequences]
print('indexed sequences', indexed_sequences)
vocab_size = len(stoi) # 词向量行长度
d_model = 3
lut = nn.Embedding(vocab_size,d_model)
lut.state_dict()['weight']
# convert the sequences to a tensor
# 将sequence转化为tensor
tensor_sequences  = torch.tensor(indexed_sequences).long()
embedding = lut(tensor_sequences) # 进行匹配
print('embedding', embedding)
```

输出结果

```
tokenized sequences [['i', 'wonder', 'what', 'will', 'come', 'next'], ['this', 'is', 'a', 'basic', 'example', 'paragraph'], ['hello', 'what', 'is', 'a', 'basic', 'split']]

indexed sequences [[11, 23, 21, 22, 5, 15], [20, 13, 0, 3, 7, 17], [10, 21, 13, 0, 3, 18]]

embedding tensor([[[ 0.3408, -1.5560, -1.9805],
         [-0.0814,  0.3851, -0.6627],
         [ 1.2521, -1.3471,  1.7690],
         [ 1.4194, -0.2584,  0.2271],
         [ 0.4447,  0.4902, -0.9708],
         [-0.5985,  1.8578, -0.4236]],

        [[ 0.0246, -0.7061,  0.3169],
         [ 0.4956,  1.2468, -0.0811],
         [ 0.2099,  0.1917, -1.3613],
         [ 0.7798, -0.5529, -0.8695],
         [ 0.2796, -0.1313, -0.2061],
         [ 0.6807, -0.3493,  0.2217]],

        [[-1.0525,  0.0185,  0.3689],
         [ 1.2521, -1.3471,  1.7690],
         [ 0.4956,  1.2468, -0.0811],
         [ 0.2099,  0.1917, -1.3613],
         [ 0.7798, -0.5529, -0.8695],
         [ 0.4003,  0.7833,  0.4854]]], grad_fn=<EmbeddingBackward0>)
```

#### 2. 总结

> 1. 关键在于embedding matrix的生成原理要搞懂,上面的例子是通过nn.Embdding()模块生成的,原理可以参考上面那两篇文章
> 2. `要理解One-Hot矩阵通过与Embedding矩阵相乘`,转化为低维的矩阵

### 1.2.2 Positional Encoding(位置编码)

> 参考这篇文章，已经讲的很清楚
>
> [【Transformer系列】深入浅出理解Positional Encoding位置编码](https://blog.csdn.net/m0_37605642/article/details/132866365)

#### 代码实现

> 下面来讲代码实现:
>
> 代码实现参考这篇文章:
>
> [Positional Encoding](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)

```python


'''
Author: Jean_Leung
Date: 2024-09-06 16:15:20
LastEditors: Jean_Leung
LastEditTime: 2024-09-06 17:57:42
FilePath: \LinearModel\transformer_practise\_positional_encoding.py
Description: 位置编码练习
                1.相对位置编码:
                    1.1 正余弦位置编码
                    1.2 复数函数位置编码
                2.绝对位置编码:

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''

# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# visualization packages
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# 建立词库
# -------------------词典序列-------------------
# 词典包含了所有所有接下来准备出现的单词
example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

# 将词库中所有的词进行tokenize

def tokenize(sequence):
    # remove punctuation
    # 移除标点符号
    for punc in["!", ".", "?",","]:
        sequence = sequence.replace(punc, "")
    # 返回一个字符数组
    # print('sequence:', sequence)
    return [token.lower() for token in sequence.split(" ")]

# 返回了tokenize之后的数组
# tokens = tokenize(example)
# print('tokens:',tokens)

# 对词典进行排序，且进行world:index映射
def build_vocab(data):
    vocab = list(set(tokenize(data))) # 先用集合将序列数组进行去重，然后再用list集合存储
    vocab.sort() # 排序
    # 建立world:index的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

# 间接建立了one-hot矩阵，因为不是用矩阵存储的，所以说是间接
stoi = build_vocab(example)
# print("stoi: ", stoi)
#

#------------------测试序列-----------------------
# sequence = [stoi[word] for word in tokenize("I wonder what will come next!")]
sequences = ["I wonder what will come next!",
             "This is a basic example paragraph.",
             "Hello, what is a basic split?"]

# 得到了tokenized的序列
def tokenize_sequence(sequences, stoi):
    # sequences_tokens: [['i', 'wonder', 'what', 'will', 'come', 'next'], ['this', 'is', 'a', 'basic', 'example', 'paragraph'], ['hello', 'what', 'is', 'a', 'basic', 'split']]
    sequences_tokens = [tokenize(sequence) for sequence in sequences] # 得到sequence token的数组例如
    # sequences_indices:  [[11, 23, 21, 22, 5, 15], [20, 13, 0, 3, 7, 17], [10, 21, 13, 0, 3, 18]]
    # 得到sequences里面的词的位置向量，是一个(3,6)的矩阵
    sequences_indices = [[stoi[word] for word in sequence_tokens] for sequence_tokens in sequences_tokens]
    return sequences_indices

# sequences_tokens = [tokenize(sequence) for sequence in sequences] # 得到sequence token的数组例如
# # print('sequences_tokens:' ,sequences_tokens)
# sequences_indices = [[stoi[word] for word in sequence_tokens] for sequence_tokens in sequences_tokens]
# print('sequences_indices: ',sequences_indices)

vocab_size = len(stoi) # 词向量行长度
d_model = 4 # embedding 的d_model维度
max_length = 10 # maximum sequence length
n = 100

# -----------------embedding------------------
# # 调用nn里面的embdding模块
# lut = nn.Embedding(vocab_size,d_model) # look-up model
# lut.state_dict()['weight']
# # for param in lut.state_dict():
# #     print('parameter',param)
# # 根据官方文档，接受的是张量，所以我们需要将sequence_indices转化为张量
# sequences_indices = tokenize_sequence(sequences=sequences,stoi=stoi)
# tensor_sequences = torch.tensor(sequences_indices).long() #转化为张量在输入到nn.Embedding里面，且需要long化
# embedding = lut(tensor_sequences) # 得到embedding矩阵
# print('embdding:',embedding)
def embedding_(vocab_size,d_model):
    lut = nn.Embedding(vocab_size,d_model) # look-up model，里面是基于word2Vec的思想进行生成矩阵
    # 得仔细研读
    lut.state_dict()['weight'] # 字典对象筛选出来，只剩tensor对象
    sequences_indices = tokenize_sequence(sequences=sequences,stoi=stoi)
    tensor_sequences = torch.tensor(sequences_indices).long() #转化为张量在输入到nn.Embedding里面，且需要long化
    embedding = lut(tensor_sequences) # 得到embedding矩阵
    return embedding
# 生成embedding之后的矩阵
embeddings = embedding_(vocab_size=vocab_size, d_model=d_model)
# print('embeddings: ',embeddings)


# -----------------相对位置编码------------------
# set the output to 2 decimal places without scientific notation
torch.set_printoptions(precision=2, sci_mode=False)

def gen_pe(max_length,d_model,n):
    # 初始化pe,将pe初始化成(max_length,d_model)的零矩阵张量
    pe = torch.zeros(max_length, d_model)
    position = torch.arange(0, max_length)
    # print('position.shape: ',position.shape)
    # print('position without unsqueeze: ',position)
    position = position.unsqueeze(-1)
    # print('position.shape: ',position.shape)
    # print('position with unsqueeze: ', position)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # add a dimension
    # 增加一个维度
    pe = pe.unsqueeze(0)
    # the output has a shape of (1, max_length, d_model)
    return pe

# encodings = gen_pe(max_length=max_length, d_model=d_model, n=n)

# print('encodings:', encodings)
# print('encodings.shape:', encodings.shape)
# select the first six tokens
# seq_length = embeddings.shape[1]
# print('encodings: ',encodings[:,:seq_length])
# print(embeddings + encodings[:, :seq_length])
# encodings[:seq_length]
# print('seq_length', seq_length)
# print(encodings[:seq_length])


#------------------使用pytorch里面的框架-----------------------
# 下面的代码是官方文档的代码
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()     

    # initialize dropout                  
    self.dropout = nn.Dropout(p=dropout)      

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x: Tensor):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 

    # perform dropout
    return self.dropout(x)

pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_length=max_length)
# print(pe.state_dict())
# print(pe(embeddings))
pe(embeddings)

```

#### 总结

> 

### 1.2.3 Encoder(编码器)

> ![img](file:///C:/Users/25212/OneDrive/Encoder和decoder区别.png)
>
> Encoder包括:
>
> - Multi-Head Attention(多头注意力机制)
> - residual connection(残差连接)
> - layer normalization(层归一化)
> - Feed Forward connection (全反馈神经网络连接)

#### 1. Multi-Head Attention

> 参照前面Self-Attention那一章

##### 代码实现



##### 总结

#### 2. residual connection(残差连接)

##### 代码实现

##### 总结

#### 3.layer normalization(层归一化)

##### 代码实现

##### 总结

#### 4.FNN(全连接前馈神经网络)

##### 代码实现

##### 总结

