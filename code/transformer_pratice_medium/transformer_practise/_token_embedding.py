'''
Author: Jean_Leung
Date: 2024-09-06 10:10:58
LastEditors: Jean_Leung
LastEditTime: 2024-09-06 21:21:05
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
embedding = lut(tensor_sequences)
print('embedding', embedding)

# Positional Encoding layer
# # vocab size
# # 获得 vocab_size
# vocab_size = len(stoi)
# # d_model 隐藏单元的个数
# d_model = 3 # 数据是三维

# # 构建一个position的embedding
# # embedding = torch.rand(vocab_size,d_model) # size为(24,3)矩阵，因为需要进行运算
# lut  = nn.Embedding(vocab_size,d_model)
# lut.state_dict()['weight'] # 返回字典dict
# indices = torch.Tensor(sequence).long()
# embedding = lut(indices)
# print(embedding)
# # embedded_sequence = embedding[sequence]
# print("embedded_sequence:",embedded_sequence)
# print("embedded_sequence.shape:",embedded_sequence.shape) # size为(6,3)
# 3D visualization of the positional encoding
# x,y,z = embedded_sequence[:,0],embedded_sequence[:,1],embedded_sequence[:,2]
# ax = plt.axes(projection='3d')
# ax.scatter3D(x,y,z)
# plt.show()