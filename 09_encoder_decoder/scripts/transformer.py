import math
import torch
from torch import nn

from attention_mechanism import DotProductAttention, AttentionDecoder
from encoder_decoder import Encoder

print("############ Transformer和RNN ############")
'''
    1、全局：RNN长距离依赖，Transformer自注意力任意两个词直接相连，能够一眼看全局
    2、计算：RNN串行计算复杂度随序列长度线性增加，Transformer大矩阵并行一次算完
    3、稳定：RNN训练深层网络极其困难，Transformer使用残差连接和层归一化网络可以无限堆叠
'''

print("############ 自注意力机制 ############")
'''
    自注意力将 {x(i)} 既当作query，又当作（key，value）来对序列抽取特征
    好处是完全并行，最长序列为1，但对长序列计算复杂度高
'''

print("############ 位置编码 ############")
'''
    给 Transformer 注入序列顺序信息。Transformer 没有 RNN 那样的时序处理，需要显式告诉模型"第几个词"
    位置编码矩阵 P，使用 X+P 作为自注意力输入（是相加而不是concat）
    位置编码使用sin和cos函数
'''

class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.0, max_len=1000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        # Step 1：生成位置矩阵
        X = torch.arange(max_len).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2) / num_hiddens)
        # Step 2：交替 sin/cos
        self.P[:, :, 0::2] = torch.sin(X)  # 偶数维度：sin
        self.P[:, :, 1::2] = torch.cos(X)  # 奇数维度：cos

    def forward(self, X):
        '''
            把位置信息编码成一串波，加到词向量上，让并行处理的 Transformer 知道词的先后顺序。

            词嵌入:   [0.1, -0.3, 0.5, ...]  ← 词的含义
            位置编码: [0.0, 1.0, 0.8, ...]  ← 词的位置
                 ↓ 相加
            输出:     [0.1, 0.7, 1.3, ...]  ← 既知道是什么词，又知道在第几个位置
        '''
        # Step 3：加到词嵌入上
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


print("############ Transformer架构 ############")
'''
    基于编码器-解码器架构来处理序列，编码器-解码器都有n个transformer块
    跟seq2seq不同，不再有RNN而是纯注意力
    
    多头注意力 Multi-head
        有多个Attention来接收 q/k/v
        多头注意力 = 多个专家同时看同一句话，各自关注不同的方面，最后把所有人的意见合并
    
    带掩码的多头注意力 Masked Multi-head
        解码器不应该能看到当前元素之后的元素，可通过valid_len来隐藏
        
    基于位置的前馈网络 Position Wise FFN
        论文强调这个 MLP 是逐位置的（Position-wise）每个词独立过同一个网络，不像注意力那样词之间交互
        
    层归一化 Add & Norm
        Add：残差 Residual 块
        Norm：LayerNorm 的 normalized_shape 参数要求的是元组或列表，表示对哪些维度做归一化
        
    编码器输出作为解码器中多头注意力的key和value，解码器连接处用交叉注意力，其余使用自注意力
'''


def transpose_qkv(x, num_heads):
    """拆成多头"""
    # 输入: (batch, seq_len, 256)
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    # → (batch, seq_len, 4, 64)  ← 4个头，每个64维

    x = x.permute(0, 2, 1, 3)
    # → (batch, 4, seq_len, 64)

    return x.reshape(-1, x.shape[2], x.shape[3])
    # → (batch*4, seq_len, 64)  ← 当成 batch × 4 个独立的样本

def transpose_output(x, num_heads):
    """合并回头"""
    # 输入: (batch*num_heads, seq_len, 64)
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    # → (batch, 4, seq_len, 64)

    x = x.permute(0, 2, 1, 3)
    # → (batch, seq_len, 4, 64)

    return x.reshape(x.shape[0], x.shape[1], -1)
    # → (batch, seq_len, 256)  ← 恢复原形状

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)

        # 1. 线性变换
        self.W_q = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 2. 拆成多头 (batch*num_heads, seq_len, num_hiddens)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # 3. 有效长度也复制num_heads份
        if valid_lens is not None:
            # 如果是二维 (batch, seq)，取每个样本的实际长度
            if valid_lens.dim() == 2:
                valid_lens = valid_lens.sum(dim=1)  # 或 valid_lens[:, -1]
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # 4. 并行计算注意力（用矩阵乘法替代循环，GPU一次算完)
        outputs = self.attention(queries, keys, values, valid_lens)

        # 5. 合并回头（输入输出shape不变）
        output_concat = transpose_output(outputs, self.num_heads)

        # 6. 最终线性变换
        return self.W_o(output_concat)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))

class AddNorm(nn.Module):
    """残差连接，防止深层网络退化"""
    def __init__(self, normlized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(normlized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)

class EncoderBlock(nn.Module):
    '''
        一个 EncoderBlock = 自注意力（交流） + FFN（思考） + 残差（防退化）
    '''
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=use_bias)
        self.addnorm1 = AddNorm(normlized_shape=norm_shape, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(normlized_shape=norm_shape, dropout=dropout)

    def forward(self, x, valid_lens):
        '''
        一个 EncoderBlock 输入输出形状完全一样，所以可以无限堆叠
            x: (batch, seq_len, d_model)
                ↓ 自注意力
            y: (batch, seq_len, d_model)  ← 形状不变
                ↓ FFN
            输出: (batch, seq_len, d_model)  ← 形状不变
        '''
        y = self.addnorm1(x, self.attention(queries=x, keys=x, values=x, valid_lens=valid_lens))
        return self.addnorm2(y, self.ffn(y))

class TransformerEncoder(Encoder):
    '''
        "Beauty and a beat"
            ↓ Embedding × √128
        [0.2, -0.1, ...] × 128 维
            ↓ + PositionEncoding
        [0.2, -0.1, ...] + [0.0, 1.0, ...]  ← 有词义 + 有位置
            ↓ Block 0: 自注意力 + FFN
            ↓ Block 1: 自注意力 + FFN
            ↓ 无限堆叠 ...
        (batch=1, seq=4, dim=128)  ← 编码器输出
    '''
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens

        # Embedding 就是存储了一个 (vocab_size, embed_dim) 的权重矩阵，根据索引去取对应行，时间复杂度O(1) 查表
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, x, valid_lens, *args):

        # 词嵌入乘 √d_model 让嵌入值和位置编码的量级匹配，防止位置编码盖过词义
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(x, valid_lens)
            '''
                blk.attention          # EncoderBlock 的 MultiHeadAttention
                    .attention         # MultiHeadAttention 的 DotProductAttention
                        .attention_weights  # DotProductAttention 保存的注意力权重
            '''
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return x

class DecoderBlock(nn.Module):
    '''
        一个 DecoderBlock = 掩码自注意力（看已生成的词） + 交叉注意力（看源句子） + FFN（思考） + 三次残差（防退化）
    '''
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(normlized_shape=norm_shape, dropout=dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(normlized_shape=norm_shape, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(normlized_shape=norm_shape, dropout=dropout)

    def forward(self, x, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = x # 第一个时间步，只有当前输入
        else:
            key_values = torch.cat([state[2][self.i], x], dim=1) # 拼上之前生成的词
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = x.shape
            # 第1个词只能看自己，第2个词看前2个，...，防止看到未来
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else: # predict 预测时每次只输入一个新词，没有未来的词，不需要掩码
            dec_valid_lens = None

        '''
            x (batch, seq_len, d_model)
             │
             │  〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰
             │  ← 子层1: 掩码自注意力
             │
             ├─→ [attention1] ──→ x2 (batch, seq_len, d_model)
             │    Q=x, K/V=已生成词, mask=未来词
             │
             ├─→ [addnorm1] ──→ y (batch, seq_len, d_model)
             │    x + Dropout(x2) → LayerNorm
             │
             │  〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰
             │  ← 子层2: 交叉注意力
             │
             ├─→ [attention2] ──→ y2 (batch, seq_len, d_model)
             │    Q=y, K/V=enc_outputs
             │
             ├─→ [addnorm2] ──→ z (batch, seq_len, d_model)
             │    y + Dropout(y2) → LayerNorm
             │
             │  〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰
             │  ← 子层3: FFN
             │
             ├─→ [FFN] ──→ ffn_out (batch, seq_len, d_model)
             │    Linear → ReLU → Linear
             │
             └─→ [addnorm3] ──→ output (batch, seq_len, d_model)
                  z + Dropout(ffn_out) → LayerNorm
        '''
        x2 = self.attention1(x, key_values, key_values, dec_valid_lens)
        y = self.addnorm1(x, x2)
        y2 = self.attention2(y, enc_outputs, enc_outputs, enc_valid_lens)
        z = self.addnorm2(y, y2)
        return self.addnorm3(z, self.ffn(z)), state

class TransformerDecoder(AttentionDecoder):
    '''
        目标句子: "Je suis étudiant"
            ↓ Embedding × √d_model
        词嵌入
            ↓ + PositionEncoding
        位置感知的输入
            ↓ Block 0: 掩码自注意力 → 交叉注意力 → FFN
            ↓ Block 1: 掩码自注意力 → 交叉注意力 → FFN
            ↓ 无限堆叠 ...
            ↓ Linear(d_model → vocab_size)
        预测: (batch, seq, vocab_size)
    '''
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, x, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(x), state

    @property
    def attention_weights(self):
        return self._attention_weights

'''
    Transformer 整体数据流
    
    源句子: "Beauty and a beat"        → (1, 4)
    目标句子: "<bos> Je suis étudiant" → (1, 4)
    
    编码器:
      Embedding        (1, 4)    → (1, 4, d_model)
      × √d_model       (1, 4, d_model)
      + PositionEnc    (1, 4, d_model)
      EncoderBlock × N (1, 4, d_model)
      enc_outputs      (1, 4, d_model)  ──────────────────────┐
                                                              │
    解码器:                                                    │
      Embedding        (1, 4)    → (1, 4, d_model)            │
      × √d_model       (1, 4, d_model)                        │
      + PositionEnc    (1, 4, d_model)                        │
      DecoderBlock 0:                                         │
        掩码自注意力    (1, 4, d_model)                         │
        AddNorm        (1, 4, d_model)                        │
        交叉注意力      Q=(1,4,d_model), K/V=enc_outputs  ←←←←←←┘
        AddNorm        (1, 4, d_model)
        FFN            (1, 4, d_model)
        AddNorm        (1, 4, d_model)
      DecoderBlock 1:
        ...(同上)...
      Linear          (1, 4, d_model) → (1, 4, vocab_size)
'''















