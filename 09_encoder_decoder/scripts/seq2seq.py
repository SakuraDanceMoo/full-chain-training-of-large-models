from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import collections
import math
import matplotlib.pyplot as plt

from machine_translation import tokenize_nmt, truncate_pad, load_translate, Vocab, build_array
from encoder_decoder import Encoder, Decoder, EncoderDecoder
from attention_mechanism import Seq2SeqAttentionDecoder
from transformer import TransformerEncoder, TransformerDecoder


print("############ seq2seq ############")
'''
    序列到序列学习，使用编码器-解码器架构，最早用于机器翻译
    编码器是一个RNN，读取输入句子，可以是双向RNN
    解码器使用另一个RNN来输出

'''

print("############ BLEU ############")
'''
    衡量生成句子的好坏BLEU，最好是1
    惩罚过短的预测，长匹配给高权重
    
    BLEU = exp( min(0, 1 - len(label) / len(pred)) ) * 累乘( Pn^( 1/2^n ))

'''

print("############ 实现seq2seq的编码器-解码器 ############")

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, state = self.rnn(x)
        return output, state

class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size) # Decoder有自己的Embedding不能和Encoder共享
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout) # 输入=词嵌入+上下文
        self.dense = nn.Linear(num_hiddens, vocab_size) # vocab_size大小的分类器

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1] # 编码器返回 (output, state)，解码器只需要 state（GRU 的隐藏状态）

    def forward(self, x, state):
        # 1. 词嵌入 + 时间步前置
        x = self.embedding(x).permute(1, 0, 2)  # (batch, steps, embed) → (steps, batch, embed)

        # 2. 取编码器最后一层的隐藏状态，复制到每个时间步作为上下文
        context = state[-1].repeat(x.shape[0], 1, 1)  # (steps, batch, num_hiddens)

        # 3. 拼接当前词嵌入和上下文
        # dim=2 是特征维度，拼接两个特征向量得到一个更长的特征向量
        x_and_context = torch.cat((x, context), dim=2)  # (steps, batch, embed+num_hiddens)

        # 4. GRU 前向
        output, state = self.rnn(x_and_context, state)

        # 5. 预测每个时间步的下一个词
        output = self.dense(output).permute(1, 0, 2)  # (batch, steps, vocab_size)

        return output, state

def sequence_mask(x, valid_len, value=0):
    """把填充位置的输出屏蔽掉，确保损失计算时忽略 <pad>"""
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32, device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value # 无效位置填 value（默认 0）
    return x

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """让 <pad> 位置的损失为 0，模型只学习真实内容。"""
    '''
        unweighted_loss: [0.8, 0.3, 1.2, 2.5, 3.1]
        weights:         [1,   1,   1,   0,   0  ]
        乘积:            [0.8, 0.3, 1.2, 0,   0  ]
        
        mean(dim=1) = (0.8 + 0.3 + 1.2 + 0 + 0) / 5 = 0.46  ← 被 0 拉低了！
    '''
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none' # 返回每个样本的独立损失，而不是一个标量平均值
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# ==================== 训练 ====================
def train_seq2seq(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    for src, tgt, src_len, tgt_len in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        src_len, tgt_len = src_len.to(device), tgt_len.to(device)

        # 解码器输入是目标句去掉最后一个词（<eos>）
        dec_input = tgt[:, :-1]
        # 解码器标签是目标句去掉第一个词（<bos>）
        dec_label = tgt[:, 1:]
        # 标签的有效长度减 1
        label_valid_len = tgt_len - 1

        optimizer.zero_grad()
        output, _ = model(src, dec_input, src_len)
        loss = criterion(output, dec_label, label_valid_len).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ==================== 预测 ====================
def predict_seq2seq(model, src_sentence, src_vocab, tgt_vocab, device, num_steps=15):
    """单句翻译"""
    model.eval()
    src_tokens = tokenize_nmt([(src_sentence, '')])[0][0]
    src_indexed = [src_vocab['<bos>']] + [src_vocab[t] for t in src_tokens] + [src_vocab['<eos>']]
    enc_x = torch.tensor([truncate_pad(src_indexed, num_steps, src_vocab['<pad>'])], device=device)
    src_len = torch.tensor([len(src_indexed)], device=device)

    enc_outputs = model.encoder(enc_x, src_len)
    dec_state = model.decoder.init_state(enc_outputs, src_len)

    dec_input = torch.tensor([[tgt_vocab['<bos>']]], device=device)
    output_seq = []

    for _ in range(num_steps):
        y, dec_state = model.decoder(dec_input, dec_state)
        pred = y.argmax(dim=2)[:, -1]
        pred_token = pred.item()
        if pred_token == tgt_vocab['<eos>']:
            break
        output_seq.append(pred_token)
        dec_input = pred.unsqueeze(0)

    return ' '.join([tgt_vocab.idx_to_token[i] for i in output_seq])


# ==================== 计算 BLEU ====================
def bleu(pred_seq, label_seq, k=4):
    """计算 BLEU 分数"""
    pred_tokens, label_tokens = pred_seq.split(), label_seq.split()
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, min(k, len_pred, len_label) + 1):
        num_matches = 0
        label_ngrams = collections.Counter([' '.join(label_tokens[i:i+n]) for i in range(len_label - n + 1)])
        for i in range(len_pred - n + 1):
            ngram = ' '.join(pred_tokens[i:i+n])
            if label_ngrams[ngram] > 0:
                num_matches += 1
                label_ngrams[ngram] -= 1
        score *= (num_matches / max(1, len_pred - n + 1)) ** (0.5 ** n)

    return score


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- 加载数据 ----------
    path = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'eng_fra_songs.txt'
    pairs = load_translate(path)
    source, target = tokenize_nmt(pairs)
    print(f"句子对数: {len(pairs)}")

    # ---------- 词表 ----------
    src_vocab = Vocab(source, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(f"源词表: {len(src_vocab)}, 目标词表: {len(tgt_vocab)}")

    # ---------- 数据预处理 ----------
    num_steps = 20
    batch_size = 32

    src_array, src_valid_len = build_array(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, num_steps)
    print(f"源: {src_array.shape}, 目标: {tgt_array.shape}")

    dataset = TensorDataset(src_array, tgt_array, src_valid_len, tgt_valid_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------- 模型 ----------
    # embed_size, num_hiddens, num_layers, dropout = 64, 128, 2, 0.5
    # LR, EPOCH = 0.003, 200

    # ---------- 使用RNN的模型 ----------
    # encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    # decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    # ---------- 使用RNN + Attention的模型 ----------
    # encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    # decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    # ---------- 使用Transformer的模型 ----------
    # Transformer 不适合小数据，最小化超参数
    embed_size, num_hiddens, num_layers, dropout = 32, 64, 1, 0.3
    num_heads, ffn_num_hiddens, norm_shape = 2, 128, [num_hiddens]
    LR, EPOCH = 8e-4, 800
    encoder = TransformerEncoder(
        vocab_size=len(src_vocab),
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        num_hiddens=num_hiddens,
        norm_shape=norm_shape,
        ffn_num_inputs=num_hiddens,
        ffn_num_hiddens=ffn_num_hiddens,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    decoder = TransformerDecoder(
        vocab_size=len(tgt_vocab),
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        num_hiddens=num_hiddens,
        norm_shape=norm_shape,
        ffn_num_inputs=num_hiddens,
        ffn_num_hiddens=ffn_num_hiddens,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )

    model = EncoderDecoder(encoder, decoder).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- 训练 ----------
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = MaskedSoftmaxCELoss()
    losses = []

    for epoch in range(EPOCH):
        loss = train_seq2seq(model, dataloader, optimizer, criterion, device)
        losses.append(loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f}")


    # ---------- 最终评估 ----------
    print("\n========== 最终评估 ==========")
    total_bleu = 0
    count = 0
    for eng, fra in pairs:
        pred = predict_seq2seq(model, eng, src_vocab, tgt_vocab, device, num_steps)
        b = bleu(pred, fra)
        total_bleu += b
        count += 1
        if count <= 5:
            print(f"源: {eng}")
            print(f"预测: {pred}")
            print(f"真实: {fra}")
            print(f"BLEU: {b:.4f}\n")

    print(f"平均 BLEU: {total_bleu/count:.4f}")

    # ---------- 画图 ----------
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Seq2Seq Training Loss')
    plt.grid(True)
    curve_path = Path(__file__).resolve().parents[1] / 'curves' / 'transformer_loss.png'
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(curve_path, dpi=150)
    plt.show()
















