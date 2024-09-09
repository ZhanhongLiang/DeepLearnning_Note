

'''
Author: Jean_Leung
Date: 2024-09-06 16:15:20
LastEditors: Jean_Leung
LastEditTime: 2024-09-06 22:36:00
FilePath: \LinearModel\transformer_practise\_positional_encoding.py
Description: 位置编码练习
                1.相对位置编码:
                    1.1 正余弦位置编码
                    1.2 复数函数位置编码
                2.绝对位置编码:

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''

'''
从_positional_encoddings来看，加和之后的shape为[batch_size, seq_length, d_model]
output: tensor([[[ 0.13,  1.53, -0.42,  0.00, -0.92,  1.85,  1.10,  0.81],
         [ 1.15, -0.74,  0.47,  0.44,  0.40,  0.76,  1.19, -1.00],
         [ 1.42,  0.21, -0.66,  1.76, -0.57,  2.84, -1.11,  0.00],
         [ 0.68, -0.58,  0.22,  0.00, -0.00,  3.04,  0.37,  1.44],
         [-0.65, -0.82,  0.14,  0.53, -0.25,  1.50, -1.03,  1.14],
         [-1.83,  0.94,  0.96,  1.04,  1.33,  0.00, -2.22,  0.13]],

        [[-1.17,  1.90,  2.02,  0.25,  2.15,  1.02, -0.46,  2.67],
         [ 0.81, -1.79,  0.00,  3.41,  0.95,  0.53,  0.01,  3.42],
         [ 0.00, -1.20,  0.80,  2.66, -0.08,  1.06, -0.90, -1.10],
         [ 0.52,  0.68, -0.20,  0.47,  0.31,  1.49, -0.12,  2.17],
         [-1.10,  0.71,  0.42, -0.18,  0.16,  0.42, -0.35,  3.84],
         [ 0.09, -0.01, -1.82,  0.00,  1.37, -0.50, -0.00,  0.89]],

        [[ 0.72,  2.71, -1.21,  1.30, -0.99, -1.24,  0.48,  0.27],
         [ 1.34,  1.27, -0.00,  1.78, -0.00,  2.84, -1.11,  0.90],
         [ 0.89, -2.86,  0.42,  3.39,  0.96,  0.53,  0.01,  3.42],
         [ 1.34, -1.84,  0.91,  2.64, -0.07,  1.06, -0.89, -1.10],
         [-0.48,  1.06, -0.09,  0.43,  0.32,  1.49, -0.12,  2.17],
         [-2.68, -0.63, -0.60,  2.73,  0.00,  0.82,  0.45,  1.44]]],
       grad_fn=<MulBackward0>)
output_.shape: torch.Size([3, 6, 8])
batch_size 为序列个数，也就是输入了几个序列，从上个程序来看，我们输入了3句话
seq_length为序列里面的词个数，我们输入的三句话都是6个词
d_model 这个是embedding matrix定义的维度，上个处理程序定义了8维
总结来看,[3,6,8]就是这么得出来的
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

# # 非继承写法
# class MyEmbedding:
    
#     def __init__(self, vocab,d_model,sequence):
#         self.vocab = vocab # 词典实例化
#         self.d_model = d_model # d_model
#         self.sequence = sequence # sequence输入的匹配序列
#         # self.embedding = nn.Embedding(len(vocab), d_model)
#     def tokenize(self):
#            # remove punctuation
#         # 移除标点符号
#         for punc in["!", ".", "?",","]:
#             self.sequence = self.sequence.replace(punc, "")
#         # 返回一个字符数组
#         # print('sequence:', sequence)
#         return [token.lower() for token in self.sequence.split(" ")]
#     def build_vocab(self):
#         vocab_ = list(set(tokenize())) # 先用集合将序列数组进行去重，然后再用list集合存储
#         vocab_.sort() # 排序
#         # 建立world:index的映射
#         word_to_idx = {word: idx for idx, word in enumerate(vocab_)}
#         return word_to_idx
#     def get_stoi(self):
#         self.stoi = build_vocab()
        
#     def tokenize_sequence(self,sequences):
#         self.sequences = sequences
#         # sequences_tokens: [['i', 'wonder', 'what', 'will', 'come', 'next'], ['this', 'is', 'a', 'basic', 'example', 'paragraph'], ['hello', 'what', 'is', 'a', 'basic', 'split']]
#         sequences_tokens = [tokenize(sequence) for sequence in sequences] # 得到sequence token的数组例如
#         # sequences_indices:  [[11, 23, 21, 22, 5, 15], [20, 13, 0, 3, 7, 17], [10, 21, 13, 0, 3, 18]]
#         # 得到sequences里面的词的位置向量，是一个(3,6)的矩阵
#         sequences_indices = [[self.stoi[word] for word in sequence_tokens] for sequence_tokens in sequences_tokens]
#         return sequences_indices
    
#     def embedding_(self,vocab_size,d_model):
#         lut = nn.Embedding(vocab_size,d_model) # look-up model
#         lut.state_dict()['weight'] # 字典对象筛选出来，只剩tensor对象
#         sequences_indices = tokenize_sequence(sequences=sequences,stoi=stoi)
#         tensor_sequences = torch.tensor(sequences_indices).long() #转化为张量在输入到nn.Embedding里面，且需要long化
#         embedding = lut(tensor_sequences) # 得到embedding矩阵
#         return embedding

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
d_model = 8 # embedding 的d_model维度
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
    lut = nn.Embedding(vocab_size,d_model) # look-up model
    lut.state_dict()['weight'] # 字典对象筛选出来，只剩tensor对象
    sequences_indices = tokenize_sequence(sequences=sequences,stoi=stoi)
    tensor_sequences = torch.tensor(sequences_indices).long() #转化为张量在输入到nn.Embedding里面，且需要long化
    embedding = lut(tensor_sequences) # 得到embedding矩阵
    return embedding

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
output_ = pe(embeddings)
print('output:', output_)
print('output_.shape:', output_.shape)

