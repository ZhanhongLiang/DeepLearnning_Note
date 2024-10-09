import torch.nn.functional as F
import torch
import numpy as np
# input是输入src tensor张量
#src为[src_len,batch_size]
# 里面有隐藏的维度是vocab_szie
input = torch.tensor([[1,2,4,5],[4,3,2,8]])
print("input_shape:",input.shape)
# embedding_matrix 是
embedding_matrix = embedding_matrix = torch.tensor(
        [[0.7330, 0.9718, 0.9023],
        [0.7769, 0.7640, 0.3664],
        [0.6036, 0.3873, 0.5681],
        [0.8422, 0.6275, 0.5400],
        [0.0346, 0.5622, 0.2547],
        [0.4926, 0.9282, 0.1762],
        [0.0037, 0.5831, 0.4443],
        [0.2001, 0.1086, 0.0518],
        [0.6574, 0.9185, 0.3451]])
# [vocab_size,embed_dim]
print(embedding_matrix.shape)
# Output = [src_len,batch_size] x [vocab_size,embed_dim] = [src_len,batch_size,embed_dim]
print(F.embedding(input,embedding_matrix))
