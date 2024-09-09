'''
Author: Jean_Leung
Date: 2024-09-06 21:13:19
LastEditors: Jean_Leung
LastEditTime: 2024-09-07 19:00:15
FilePath: \LinearModel\transformer_practise\_feed_forward_network.py
Description: 从零开始实现Multi-Head Attention
                1. 

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
'''

'''
上个模块positional_encodding最后得出的matrix是[batch_size, seq_length, d_model]=[3,6,8]
这部分继续作为Tensor继续输入到Multi_Head Attention中
接下来继续探讨经过Multi_Head Attention过后的维度

'''


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

# 可视化代码
def display_attention(sentence: list, translation: list, attention: Tensor, 
                      n_heads: int = 8, n_rows: int = 4, n_cols: int = 2):
  """
    Display the attention matrix for each head of a sequence.

    Args:
        sentence:     German sentence to be translated to English; list
        translation:  English sentence predicted by the model
        attention:    attention scores for the heads
        n_heads:      number of heads
        n_rows:       number of rows
        n_cols:       number of columns
  """
  # ensure the number of rows and columns are equal to the number of heads
  assert n_rows * n_cols == n_heads
    
  # figure size
  fig = plt.figure(figsize=(15,20))
    
  # visualize each head
  for i in range(n_heads):
        
    # create a plot
    ax = fig.add_subplot(n_rows, n_cols, i+1)
        
    # select the respective head and make it a numpy array for plotting
    _attention = attention.squeeze(0)[i,:,:].cpu().detach().numpy()

    # plot the matrix
    cax = ax.matshow(_attention, cmap='bone')

    # set the size of the labels
    ax.tick_params(labelsize=12)

    # set the indices for the tick marks
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(translation)))

    ax.set_xticklabels(sentence)
    ax.set_yticklabels(translation)

  plt.show()

# print('output:', output_)
# print('output_.shape:', output_.shape)


# ----------------------Multi_head_attention多头注意力模型-------------------

 
Wq = nn.Linear(d_model,d_model) # query weights (8,8)
Wk = nn.Linear(d_model, d_model)          # key weights   (8,8)
Wv = nn.Linear(d_model, d_model)          # value weights (8,8)
# Wq、Wk、Wv是key
# print(Wq.state_dict())
# Wq.state_dict()['weight']
# Q K V 是 (batch_size, seq_length, d_model) 
Q = Wq(output_) # (3,6,8)x(broadcast 8,8) = (3,6,8)
K = Wk(output_) # (3,6,8)x(broadcast 8,8) = (3,6,8)
V = Wv(output_) # (3,6,8)x(broadcast 8,8) = (3,6,8)

# print('Q:',Q)
# print('K',K)
# print('V',V)


'''
# d_key = (d_model / n_heads)
# 下一步就是d_model / n_heads多头就是这么来的
# Q K V 是 (batch_size, seq_length, d_model) → (batch_size, seq_length, n_heads, d_key)变成多头Q(1,1),Q(1,2),Q(1,3),Q(1,4)
'''
batch_size = Q.shape[0]
n_heads = 4
d_key = d_model//n_heads # 8/4 = 2
# query tensor | -1 = query_length (3, 6, 8) -> (3, 6, 4, 2)
# (batch_size, seq_length, n_heads, d_key)
Q = Q.view(batch_size, -1, n_heads, d_key)
# value tensor | -1 = key_length | (3, 6, 8) -> (3, 6, 4, 2) 
# (batch_size, seq_length, n_heads, d_key)
K = K.view(batch_size,-1,n_heads,d_key)
# value tensor | -1 = key_length | (3, 6, 8) -> (3, 6, 4, 2) 
# (batch_size, seq_length, n_heads, d_key)
V = V.view(batch_size,-1,n_heads,d_key)

# query tensor | (3, 6, 4, 2) -> (3, 4, 6, 2) 
# (batch_size, n_heads, seq_length, d_key)
Q = Q.permute(0, 2, 1, 3)
# key tensor | (3, 6, 4, 2) -> (3, 4, 6, 2)
# (batch_size, n_heads, seq_length, d_key)
K = K.permute(0, 2, 1, 3)
# value tensor | (3, 6, 4, 2) -> (3, 4, 6, 2) 
# (batch_size, n_heads, seq_length, d_key)
V = V.permute(0, 2, 1, 3)

# calculate scaled dot product
# (batch_size, n_heads, Q_length, d_key) x (batch_size, n_heads, d_key, K_length) = (batch_size, n_heads, Q_length, K_length)
scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(d_key) # (batch_size, n_heads, Q_length, K_length)
# apply softmax to get context for each token and others
attn_probs = torch.softmax(scaled_dot_prod, dim=-1) # (batch_size, n_heads, Q_length, K_length)

# sequence 0
# display_attention(["i", "wonder", "what", "will", "come", "next"], 
#                   ["i", "wonder", "what", "will", "come", "next"], 
#                   attn_probs[0], 4, 2, 2)

# # sequence 1
# display_attention(["this", "is", "a", "basic", "example", "paragraph"], 
#                   ["this", "is", "a", "basic", "example", "paragraph"], 
#                   attn_probs[1], 4, 2, 2)

# # sequence 2
# display_attention(["hello", "what", "is", "a", "basic", "split"], 
#                   ["hello", "what", "is", "a", "basic", "split"], 
#                   attn_probs[2], 4, 2, 2)

# multiply attention and values to get reweighted values
A = torch.matmul(attn_probs, V) # (batch_size, n_heads, Q_length, d_key)

# transpose from (3, 4, 6, 2) -> (3, 6, 4, 2)
A = A.permute(0, 2, 1, 3).contiguous()

# reshape from (3, 6, 4, 2) -> (3, 6, 8) = (batch_size, Q_length, d_model)
A = A.view(batch_size, -1, n_heads*d_key)

Wo = nn.Linear(d_model, d_model)

# (3, 6, 8) x (broadcast 8, 8) = (3, 6, 8)
output = Wo(A)    

#--------------------------FNN神经网络---------------
# d_ffn = d_model * 4 # 32
# w_1 = nn.Linear(d_model,d_ffn) # (8, 32)
# w_2 = nn.Linear(d_ffn, d_model) # (32, 8)
# # (3, 6, 8) x (8, 32) → (3, 6, 32)
# ffn_1 = w_1(output).relu()
# # print("ffn_1 :", ffn_1)
# # (3, 6, 32) x (32, 8) = (3, 6, 8)
# ffn_2 = w_2(ffn_1)

class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()

    self.w_1 = nn.Linear(d_model, d_ffn)
    self.w_2 = nn.Linear(d_ffn, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    """
    Args:
        x:            output from attention (batch_size, seq_length, d_model)
       
    Returns:
        expanded-and-contracted representation (batch_size, seq_length, d_model)
    """
    # w_1(x).relu(): (batch_size, seq_length, d_model) x (d_model,d_ffn) -> (batch_size, seq_length, d_ffn)
    # w_2(w_1(x).relu()): (batch_size, seq_length, d_ffn) x (d_ffn, d_model) -> (batch_size, seq_length, d_model) 
    return self.w_2(self.dropout(self.w_1(x).relu()))

# calculate the d_ffn
d_ffn = d_model * 4 # 32
# pass the tensor through the position-wise feed-forward network
ffn = PositionwiseFeedForward(d_model, d_ffn, dropout=0.1)
ffn(output)



