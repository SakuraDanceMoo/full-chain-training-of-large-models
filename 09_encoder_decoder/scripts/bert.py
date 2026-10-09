import math
import torch
from torch import nn

from transformer import EncoderBlock


print("############ BERT ############")
'''
    BERT动机：基于微调（迁移学习）的NLP模型
    预训练模型抽取了足够多的信息，新任务只需要更改最后的输出层
    
    BERT架构：只有编码器的Transformer，模型更大数据更多
    
    对输入的修改
        1、每个样本都是一个句子对 
            <cls> the movie is great <seq> i like it <seq>
        2、加入额外的片段Segment Embeddings
            用于区分source和target句子
        3、位置编码Position Embeddings可学习
            不再使用sin和cos
'''

print("############ 预训练任务 ############")
'''
    1、带掩码的语言模型
        Transformer的编码器是双向，但语言模型预测任务要求单向
        每次随机（15% 概率）将一些词换成<mask>，这就变成了预测完形填空
    2、判断句子是否相邻
        下一个句子预测任务，判断句子 B 是不是句子 A 的下一句
        训练样本中一半句子对是相邻，一半句子对随机拼
'''


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """给两个句子构造bert的输入"""
    tokens = ['<cls>'] + tokens_a + ['<seq>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<seq>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nn.Module):
    '''
        BERT 编码器 = Transformer 编码器 + 三合一嵌入层
        核心的创新是段嵌入 和 可学习位置编码
    '''
    def __init__(self, vocab_size, num_hiddens, norm_shape,
                 ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=1000,
                 key_size=768, query_size=768, value_size=768, use_bias=False, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)

        # Embedding 就是存储了一个 (vocab_size, embed_dim) 的权重矩阵，根据索引去取对应行，时间复杂度O(1) 查表
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, tokens, segments,  valid_lens):
        '''
            完整数据流
            输入: tokens=[CLS, I, love, SEP, You, love, SEP]
               segments=[0, 0, 0, 0, 1, 1, 1]

            token_emb:   [E_CLS, E_I, E_love, E_SEP, E_You, E_love, E_SEP]
            segment_emb: [E_A,   E_A, E_A,    E_A,   E_B,   E_B,    E_B  ]
            pos_emb:     [P_0,   P_1, P_2,    P_3,   P_4,   P_5,    P_6  ]
                              ↓ 三者相加
            x:           [v0, v1, v2, v3, v4, v5, v6]
                              ↓ EncoderBlock × N
            输出:        (batch, seq, num_hiddens)
        '''
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = x + self.pos_embedding.data[:, :x.shape[1]]
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x

class MaskLM(nn.Module):
    """1、带掩码的语言模型"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size),
        )

    def forward(self, x, pred_positions):
        '''
            输入句子: "I [MASK] you [MASK]"
                          ↑           ↑
                       位置1被遮    位置3被遮

            BERT 编码器输出:
            x: (batch=1, seq=4, 768)
               [v_I, v_MASK1, v_you, v_MASK2]

            pred_positions = [[1, 3]]  ← 被遮的位置索引
            num_pred_positions = 2     ← 遮了 2 个词

            提取被遮位置:
            batch_idx = [0, 0]
            pred_positions = [1, 3]
            masked_x = [x[0,1], x[0,3]]  ← 两个被遮位置的向量
                     → reshape(1, 2, 768)

            MLP 输入: (batch=1, num_masks=2, 768)
            MLP 输出: (1, 2, vocab_size)  ← 预测两个被遮的词
        '''
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_x = x[batch_idx, pred_positions]
        masked_x = masked_x.reshape((batch_size, num_pred_positions))
        mlm_y_hat = self.mlp(masked_x)
        return mlm_y_hat

class NextSentencePred(nn.Module):
    """2、判断句子是否相邻（2分类）"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        '''
            只取 [CLS] 位置，因为 [CLS] 已经通过多层 BERTEncoder 编码了整个句子对的信息
            
            自注意力让 [CLS] 逐步聚合信息:
                浅层: [CLS] ← {I, love}
                中层: [CLS] ← {I, love, you, SEP}
                深层: [CLS] ← {I, love, you, SEP, You, love, me, SEP}
        '''
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, x):
        return self.output(x)

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape,
                 ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=1000,
                 key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        '''
            组件	                    类型	                    输入	                输出	        作用
            self.encoder	        BERTEncoder	            tokens, segments	encoded_x	编码整个句子对
            self.mlm	            MaskLM	                被遮位置的向量	    mlm_y_hat	预测被遮住的词
            self.hidden + self.nsp	Linear+Tanh + Linear	[CLS] 的向量	        nsp_y_hat	预测是否下一句
        '''
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len, key_size, query_size, value_size)
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_x = self.encoder(tokens, segments, valid_lens)

        # 2. 掩码语言模型（训练时才做）
        if pred_positions is not None:
            mlm_y_hat = self.mlm(encoded_x, pred_positions)
        else:
            mlm_y_hat = None # 预测时不需要

        # 3. 下一句预测（取 [CLS] 位置）
        nsp_y_hat = self.nsp(self.hidden(encoded_x[:,0,:]))
        return encoded_x, mlm_y_hat, nsp_y_hat











