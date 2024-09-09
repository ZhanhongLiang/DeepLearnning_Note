

'''
Author: Jean_Leung
Date: 2024-09-06 21:13:19
LastEditors: Jean_Leung
LastEditTime: 2024-09-07 19:00:15
FilePath: \LinearModel\transformer_practise\_feed_forward_network.py
Description: 
            1.编码器

Copyright (c) 2024 by ${robotlive limit}, All Rights Reserved. 
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
from torch.nn.functional import pad



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

# vocab_size = len(stoi) # 词向量行长度
# d_model = 8 # embedding 的d_model维度
# max_length = 10 # maximum sequence length
# n = 100

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

# embeddings = embedding_(vocab_size=vocab_size, d_model=d_model)
# print('embeddings: ',embeddings)

class Embeddings(nn.Module):
  def __init__(self, vocab_size: int, d_model: int):
    """
    Args:
      vocab_size:     size of vocabulary
      d_model:        dimension of embeddings
    """
    # inherit from nn.Module
    super().__init__()   
     
    # embedding look-up table (lut)                          
    self.lut = nn.Embedding(vocab_size, d_model)   

    # dimension of embeddings 
    self.d_model = d_model                          

  def forward(self, x: Tensor):
    """
    Args:
      x:              input Tensor (batch_size, seq_length)
      
    Returns:
                      embedding vector
    """
    # embeddings by constant sqrt(d_model)
    return self.lut(x) * math.sqrt(self.d_model)  

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

 
# Wq = nn.Linear(d_model,d_model) # query weights (8,8)
# Wk = nn.Linear(d_model, d_model)          # key weights   (8,8)
# Wv = nn.Linear(d_model, d_model)          # value weights (8,8)
# Wq、Wk、Wv是key
# print(Wq.state_dict())
# Wq.state_dict()['weight']
# Q K V 是 (batch_size, seq_length, d_model) 
# Q = Wq(output_) # (3,6,8)x(broadcast 8,8) = (3,6,8)
# K = Wk(output_) # (3,6,8)x(broadcast 8,8) = (3,6,8)
# V = Wv(output_) # (3,6,8)x(broadcast 8,8) = (3,6,8)

# print('Q:',Q)
# print('K',K)
# print('V',V)


'''
# d_key = (d_model / n_heads)
# 下一步就是d_model / n_heads多头就是这么来的
# Q K V 是 (batch_size, seq_length, d_model) → (batch_size, seq_length, n_heads, d_key)变成多头Q(1,1),Q(1,2),Q(1,3),Q(1,4)
'''
# batch_size = Q.shape[0]
# n_heads = 4
# d_key = d_model//n_heads # 8/4 = 2
# # query tensor | -1 = query_length (3, 6, 8) -> (3, 6, 4, 2)
# # (batch_size, seq_length, n_heads, d_key)
# Q = Q.view(batch_size, -1, n_heads, d_key)
# # value tensor | -1 = key_length | (3, 6, 8) -> (3, 6, 4, 2) 
# # (batch_size, seq_length, n_heads, d_key)
# K = K.view(batch_size,-1,n_heads,d_key)
# # value tensor | -1 = key_length | (3, 6, 8) -> (3, 6, 4, 2) 
# # (batch_size, seq_length, n_heads, d_key)
# V = V.view(batch_size,-1,n_heads,d_key)

# # query tensor | (3, 6, 4, 2) -> (3, 4, 6, 2) 
# # (batch_size, n_heads, seq_length, d_key)
# Q = Q.permute(0, 2, 1, 3)
# # key tensor | (3, 6, 4, 2) -> (3, 4, 6, 2)
# # (batch_size, n_heads, seq_length, d_key)
# K = K.permute(0, 2, 1, 3)
# # value tensor | (3, 6, 4, 2) -> (3, 4, 6, 2) 
# # (batch_size, n_heads, seq_length, d_key)
# V = V.permute(0, 2, 1, 3)

# # calculate scaled dot product
# # (batch_size, n_heads, Q_length, d_key) x (batch_size, n_heads, d_key, K_length) = (batch_size, n_heads, Q_length, K_length)
# scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(d_key) # (batch_size, n_heads, Q_length, K_length)
# # apply softmax to get context for each token and others
# attn_probs = torch.softmax(scaled_dot_prod, dim=-1) # (batch_size, n_heads, Q_length, K_length)

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

# # multiply attention and values to get reweighted values
# A = torch.matmul(attn_probs, V) # (batch_size, n_heads, Q_length, d_key)

# # transpose from (3, 4, 6, 2) -> (3, 6, 4, 2)
# A = A.permute(0, 2, 1, 3).contiguous()

# # reshape from (3, 6, 4, 2) -> (3, 6, 8) = (batch_size, Q_length, d_model)
# A = A.view(batch_size, -1, n_heads*d_key)

# Wo = nn.Linear(d_model, d_model)

# # (3, 6, 8) x (broadcast 8, 8) = (3, 6, 8)
# output = Wo(A)    


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        n_heads:      number of self attention heads
        dropout:      probability of dropout occurring
    """
    super().__init__()
    assert d_model % n_heads == 0            # ensure an even num of heads
    self.d_model = d_model                   # 512 dim
    self.n_heads = n_heads                   # 8 heads
    self.d_key = d_model // n_heads          # assume d_value equals d_key | 512/8=64

    self.Wq = nn.Linear(d_model, d_model)    # query weights
    self.Wk = nn.Linear(d_model, d_model)    # key weights
    self.Wv = nn.Linear(d_model, d_model)    # value weights
    self.Wo = nn.Linear(d_model, d_model)    # output weights

    self.dropout = nn.Dropout(p=dropout)     # initialize dropout layer  

  def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
    """
    Args:
       query:         query vector         (batch_size, q_length, d_model)
       key:           key vector           (batch_size, k_length, d_model)
       value:         value vector         (batch_size, s_length, d_model)
       mask:          mask for decoder     

    Returns:
       output:        attention values     (batch_size, q_length, d_model)
       attn_probs:    softmax scores       (batch_size, n_heads, q_length, k_length)
    """
    batch_size = key.size(0)                  
        
    # calculate query, key, and value tensors
    Q = self.Wq(query)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)
    K = self.Wk(key)                         # (32, 10, 512) x (512, 512) = (32, 10, 512)
    V = self.Wv(value)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)

    # split each tensor into n-heads to compute attention

    # query tensor
    Q = Q.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
               -1,                           # -1 = q_length
               self.n_heads,              
               self.d_key
               ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, q_length, d_key)
    # key tensor
    K = K.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
               -1,                           # -1 = k_length
               self.n_heads,              
               self.d_key
               ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, k_length, d_key)
    # value tensor
    V = V.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
               -1,                           # -1 = v_length
               self.n_heads, 
               self.d_key
               ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, v_length, d_key)
       
    # computes attention
    # scaled dot product -> QK^{T}
    scaled_dot_prod = torch.matmul(Q,        # (32, 8, 10, 64) x (32, 8, 64, 10) -> (32, 8, 10, 10) = (batch_size, n_heads, q_length, k_length)
                                   K.permute(0, 1, 3, 2)
                                   ) / math.sqrt(self.d_key)      # sqrt(64)
        
    # fill those positions of product as (-1e10) where mask positions are 0
    if mask is not None:
      scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)

    # apply softmax 
    attn_probs = torch.softmax(scaled_dot_prod, dim=-1)
        
    # multiply by values to get attention
    A = torch.matmul(self.dropout(attn_probs), V)       # (32, 8, 10, 10) x (32, 8, 10, 64) -> (32, 8, 10, 64)
                                                        # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key) -> (batch_size, n_heads, q_length, d_key)

    # reshape attention back to (32, 10, 512)
    A = A.permute(0, 2, 1, 3).contiguous()              # (32, 8, 10, 64) -> (32, 10, 8, 64)
    A = A.view(batch_size, -1, self.n_heads*self.d_key) # (32, 10, 8, 64) -> (32, 10, 8*64) -> (32, 10, 512) = (batch_size, q_length, d_model)
        
    # push through the final weight layer
    output = self.Wo(A)                                 # (32, 10, 512) x (512, 512) = (32, 10, 512) 

    return output, attn_probs                           # return attn_probs for visualization of the scores

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

# # calculate the d_ffn
# d_ffn = d_model * 4 # 32
# # pass the tensor through the position-wise feed-forward network
# ffn = PositionwiseFeedForward(d_model, d_ffn, dropout=0.1)
# ffn(output)



# ---------------------层归一化处理---------------

class LayerNorm(nn.Module):

  def __init__(self, features, eps=1e-5):
    super().__init__()
    # initialize gamma to be all ones
    self.gamma = nn.Parameter(torch.ones(features)) 
    # initialize beta to be all zeros
    self.beta = nn.Parameter(torch.zeros(features)) 
    # initialize epsilon
    self.eps = eps

  def forward(self, src):
    # mean of the token embeddings
    mean = src.mean(-1, keepdim=True)        
    # variance of the token embeddings         
    var = src.var(-1, keepdim=True,unbiased=False)  
    # return the normalized value  
    return self.gamma * (src - mean) / torch.sqrt(var + self.eps) + self.beta 
  
#---------------------the encoder----------------------

class EncoderLayer(nn.Module):  
  def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
    """
    Args:
        d_model:      dimension of embeddings
        n_heads:      number of heads
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()
    # multi-head attention sublayer
    self.attention = MultiHeadAttention(d_model, n_heads, dropout)
    # layer norm for multi-head attention
    self.attn_layer_norm = nn.LayerNorm(d_model)

    # position-wise feed-forward network
    self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
    # layer norm for position-wise ffn
    self.ffn_layer_norm = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, src: Tensor, src_mask: Tensor):
    """
    Args:
        src:          positionally embedded sequences   (batch_size, seq_length, d_model)
        src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)
    Returns:
        src:          sequences after self-attention    (batch_size, seq_length, d_model)
    """
    # pass embeddings through multi-head attention
    _src, attn_probs = self.attention(src, src, src, src_mask)

    # residual add and norm
    src = self.attn_layer_norm(src + self.dropout(_src))
    
    # position-wise feed-forward network
    _src = self.positionwise_ffn(src)

    # residual add and norm
    src = self.ffn_layer_norm(src + self.dropout(_src)) 

    return src, attn_probs
  


class Encoder(nn.Module):
  def __init__(self, d_model: int, n_layers: int, 
               n_heads: int, d_ffn: int, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        n_layers:     number of encoder layers
        n_heads:      number of heads
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()
    
    # create n_layers encoders 
    self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffn, dropout)
                                 for layer in range(n_layers)])

    self.dropout = nn.Dropout(dropout)
    
  def forward(self, src: Tensor, src_mask: Tensor):
    """
    Args:
        src:          embedded sequences                (batch_size, seq_length, d_model)
        src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)

    Returns:
        src:          sequences after self-attention    (batch_size, seq_length, d_model)
    """

    # pass the sequences through each encoder
    for layer in self.layers:
      src, attn_probs = layer(src, src_mask)

    self.attn_probs = attn_probs

    return src
  
#----------------Why Mask Paddding------------------
def pad_seq(seq: Tensor, max_length: int = 10, pad_idx: int = 0):
  pad_to_add = max_length - len(seq) # amount of padding to add
  
  return pad(seq,(0, pad_to_add), value=pad_idx,)

#-------------------src maske过程---------------------
def make_src_mask(src: Tensor, pad_idx: int = 0):
  """
  Args:
      src:          raw sequences with padding        (batch_size, seq_length)              
    
  Returns:
      src_mask:     mask for each sequence            (batch_size, 1, 1, seq_length)
  """
  # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
  src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

  return src_mask

#--------------------test-----------------------

torch.set_printoptions(precision=2, sci_mode=False)

# convert the sequences to integers
sequences = ['What will come next?',
             'This is a basic paragraph.',
             'A basic split will come next!']

# parameters
vocab_size = len(stoi) + 1 # 找到词典的大小
d_model = 8
d_ffn = d_model*4 # 32
n_heads = 4
n_layers = 4
dropout = 0.1


# tokenize the sequences
tokenized_sequences = [tokenize(seq) for seq in sequences]

# index the sequences 
indexed_sequences = [[stoi[word] for word in seq] for seq in tokenized_sequences]
# for seq in seqs

pad_idx = len(stoi) # number of padding
max_langth = 8 # 序列里面一句话最长的长度

# 这一步是padding的过程，需要进行padding扩充
padded_seqs = []
for seq in indexed_sequences:
  padded_seqs.append(pad_seq(torch.Tensor(seq),max_langth,pad_idx))

# convert the sequences to a tensor
# tensor_sequences = torch.tensor(indexed_sequences).long()
tensor_sequences = torch.stack(padded_seqs).long()

# create the source masks for the sequences
src_mask = make_src_mask(tensor_sequences, pad_idx)

# create the embeddings
lut = Embeddings(vocab_size, d_model) # look-up table (lut)

# create the positional encodings
pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_length=10)

# embed the sequence
embeddings = lut(tensor_sequences)

# positionally encode the sequences
X = pe(embeddings)

# initialize encoder
encoder = Encoder(d_model, n_layers, n_heads,
                  d_ffn, dropout)

# pass through encoder
encoder(src=X,src_mask = src_mask)

# probabilities for sequence 0
print(encoder.attn_probs[0])
# print(encoder(src=X, src_mask=None))
# sequence 0
# # preview each sequence
for i in range(0,3):
  display_attention(tensor_sequences[i].int().tolist(), tensor_sequences[i].int().tolist(), encoder.attn_probs[i], n_heads, n_rows=2, n_cols=2)


