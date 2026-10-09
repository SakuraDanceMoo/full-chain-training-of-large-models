from pathlib import Path
import collections
import torch
from torch.utils.data import DataLoader, TensorDataset

def count_corpus(tokens):
    """统计词元的出现频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """
        原始文本 → [Tokenizer 按规则切分] → token 序列 → [Vocab 查表] → ID 序列
    
        词汇表：将词元映射到从 0 开始的数字索引中
        设计	                            原因
        <unk> 放索引 0	                未登录词自动映射到 0，不需要判断
        过滤低频词	                    减少词表大小，加速训练
        idx_to_token + token_to_idx	    O(1)双向查询
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        # 排序后高频词给小索引，低频词给大索引
        self.token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens # unk指的是unknow token
        # 词元出现次数少于min_freq则丢弃
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        '''
            建立双向映射            
            self.idx_to_token = ['<unk>', '<pad>', 'the', 'a', 'cat', ...]
            self.token_to_idx = {'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'cat': 4, ...}
        '''
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """__len__：词表大小"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """__getitem__：token → 索引"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """to_tokens：索引 → token"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]


print("############ 机器翻译数据集 ############")

# ==================== 英语→法语 数据集 ====================
def load_translate(path):
    """从 txt 文件读取英法平行语料"""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eng, fra = line.split('\t')
            pairs.append((eng.strip(), fra.strip()))
    return pairs


# ==================== 分词函数（保留标点符号作为独立 token） ====================
def tokenize_nmt(pairs, max_len=0):
    """将英法平行语料分割为 source 和 target 词元列表，标点符号作为独立 token"""
    source_tokens, target_tokens = [], []

    for eng, fra in pairs:
        # 处理英语
        eng = eng.strip()
        for punct in ['.', ',', '!', '?', ';', ':', '\'']:
            eng = eng.replace(punct, f' {punct} ')
        src = eng.split()

        # 处理法语
        fra = fra.strip()
        for punct in ['.', ',', '!', '?', ';', ':', '\'']:
            fra = fra.replace(punct, f' {punct} ')
        tgt = fra.split()

        # 截断到最大长度
        if max_len > 0:
            src = src[:max_len]
            tgt = tgt[:max_len]

        source_tokens.append(src)
        target_tokens.append(tgt)

    return source_tokens, target_tokens


# ==================== 截断和填充 ====================
def truncate_pad(line, num_steps, padding_token):
    """截断或填充句子到固定长度"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array(tokens, vocab, num_steps, max_sentences=0):
    """
    将词元列表转为 (句子数, num_steps) 的索引张量
    丢弃词数小于 2 的句子，添加 <bos> 和 <eos>
    """
    valid_lens = []
    arrays = []

    for line in tokens:
        if len(line) < 2:
            continue  # 丢弃太短的句子
        # 添加 <bos> 和 <eos>
        indexed = [vocab['<bos>']] + [vocab[t] for t in line] + [vocab['<eos>']]
        valid_lens.append(len(indexed))
        # 填充到 num_steps
        padded = truncate_pad(indexed, num_steps, vocab['<pad>'])
        arrays.append(padded)
        if max_sentences > 0 and len(arrays) >= max_sentences:
            break

    return torch.tensor(arrays), torch.tensor(valid_lens)


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 加载数据
    path = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'eng_fra.txt'
    pairs = load_translate(path)
    source, target = tokenize_nmt(pairs)
    print(f"翻译句子对数: {len(pairs)}")

    # 构建词表（你的 Vocab 类）
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    print(f"源词表大小: {len(src_vocab)}")
    print(f"目标词表大小: {len(tgt_vocab)}")
























