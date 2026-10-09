import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from encoder_decoder import Decoder


print("############ 注意力机制 ############")
'''
    注意力机制，通过主动线索query和不主动线索key来有偏向性的选择输入
    简单说就是，通过找和目标query最相近的key来获取value
    
'''

print("############ 注意力分数 ############")
'''
    注意力分数是 q 和 k 的相似度，如果 q 和 k 越相关，它们计算出的注意力分数就越高
    注意力权重是注意力分数的softmax结果
    
    类比你去图书馆找书：
    概念	        图书馆类比	            代码
    Q (Query)	你想要的："深度学习"	    Q 张量
    K (Key)	    每本书的标题	            K 张量
    V (Value)	每本书的内容	            V 张量
    Q @ K.T	    比较你的需求和每本书标题	注意力分数
    softmax	    选最匹配的几本书	        注意力权重
    @ V	        读这几本书的内容	        输出

'''

print("############ 分数的两种计算方法 ############")
'''
    f(q, (k, v)) = 累加 softmax( a(q, k(i)) ) * v(i) 
    注意力分数 a 有两种设计
        1、加性注意力：通过参数 W_q,W_k 学习什么样的 q 和 k 组合可以得高分
            a(q, k) = W_v @ tanh(W_q·Q + W_k·K)
            好处是 q 和 k 向量长度可以不相同
            相当于把 q 和 k 合并丢进单隐藏层MLP
        2、内积注意力：余弦相似度 <q, k> = cos(θ) × ||q|| × ||k||
            a(q, k) = <q, k> / 根号d 
            好处是使用矩阵乘法GPU友好，计算效率高
            q 和 k 向量长度都是d，二者作内积
            除 根号d 是为了防止内积值过大，导致 Softmax 进入梯度消失
'''


def sequence_mask(x, valid_len, value=0):
    """把填充位置的输出屏蔽掉"""
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32, device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(x, valid_lens=None):
    """
    带掩码的 softmax

    Args:
        x: (batch, queries, keys) 注意力分数矩阵
        valid_lens: (batch,) 每个样本的有效 key 数量，或 None 表示全有效

    Returns:
        softmax 后的注意力权重，<pad> 位置权重为 0
    """
    if valid_lens is None:
        return F.softmax(x, dim=-1)

    # 获取 key 的数量
    num_keys = x.shape[-1]

    # 生成掩码: (batch, 1, 1) → (batch, 1, num_keys)
    mask = torch.arange(num_keys, dtype=torch.float32, device=x.device)[None, None, :] < valid_lens[:, None, None]

    # <pad> 位置设为 -inf，softmax 后变 0
    x = x.masked_fill(~mask, float('-inf'))

    return F.softmax(x, dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        queries, keys = self.W_q(queries), self.W_k(keys)

        '''
        拼接queries和keys: (batch, queries, keys, num_hiddens) 
        
            queries.shape  # (2, 4, 8)    ← 4 个查询
            keys.shape     # (2, 6, 8)    ← 6 个键

            # 在 dim=2 插入新维度
            queries.unsqueeze(2).shape  # (2, 4, 1, 8)
            keys.unsqueeze(1).shape     # (2, 1, 6, 8)

            # 广播相加
            (2, 4, 1, 8)
            (2, 1, 6, 8)
            ─────────────────
            (2, 4, 6, 8)  ← 每个 query 和每个 key 都加了一次
        '''
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1) # (batch, queries, keys) W_v输出num_hiddens=1故去掉
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values) # 每个batch作 @ V

class DotProductAttention(nn.Module):
    """点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) # 不用学任何参数

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]

        '''
        (batch, queries, d) @ (batch, d, keys) = (batch, queries, keys)
        
            queries: (2, 4, 64)     ← 4 个查询
            keys:    (2, 6, 64)     ← 6 个键
            values:  (2, 6, 128)    ← 6 个值
            
            Q @ K.T / √64 → scores: (2, 4, 6)    ← 4×6 的相似度矩阵
                ↓ softmax + mask
            weights: (2, 4, 6)                     ← 每行和为 1
                ↓ @ V
            output: (2, 4, 128)                    ← 4 个加权后的值
        '''
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


print("############ 使用注意力的seq2seq ############")
'''
    注意力可以根据解码器RNN输出匹配到合适的编码器RNN输出来更有效的传递信息

    1、编码器输出作为key和value，解码器输出作为query，(q, (k, v)) 放入注意力
    2、注意力输出和下一个词嵌入合并进入解码器预测下一个query
'''

class AttentionDecoder(Decoder):
    """使用注意力的解码器接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """带注意力机制的 Seq2Seq 解码器。和之前固定上下文的解码器不同，这里每个时间步都动态计算上下文向量。"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens,
                                           dropout=dropout)  # q 和 k 长度相同为num_hiddens

        self.embedding = nn.Embedding(vocab_size, embed_size) # Decoder有自己的Embedding不能和Encoder共享
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout) # 输入=词嵌入+上下文
        self.dense = nn.Linear(num_hiddens, vocab_size) # vocab_size大小的分类器


    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, x, state):
        enc_outputs, hidden_state, enc_valid_lens = state

        # 1. 词嵌入 + 时间步前置
        X = self.embedding(x).permute(1, 0, 2)  # (batch, steps, embed) → (steps, batch, embed)
        outputs, self._attention_weights = [], []
        for x in X:  # x: (batch, embed) ← 当前时间步的一个词
            # 1. 用当前隐藏状态作为 query，去编码器输出中找相关位置
            query = torch.unsqueeze(hidden_state[-1], 1)  # (batch, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            #   context: (batch, 1, num_hiddens) ← 动态计算的上下文

            # 2. 拼接上下文和当前词嵌入
            x = torch.cat((context, torch.unsqueeze(x, 1)), dim=-1)
            #   (batch, 1, num_hiddens + embed_size)

            # 3. GRU 前向
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)

            # 4. 保存这一时间步的输出和注意力权重
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        outputs = self.dense(torch.cat(outputs, dim=0)) # (n, batch, num_hiddens) 时间步维度堆叠一次性通过 Linear
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


