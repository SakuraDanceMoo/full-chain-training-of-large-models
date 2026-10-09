import torch.nn as nn


print("############ 编码器-解码器 ############")
'''
    编码器：将输入编程成中间表达形式（特征）
    解码器：将中间表达形式翻译成标号
'''

# ==================== 编码器-解码器架构 ====================
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, x, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, enc_valid_lens=None, *args):
        enc_outputs = self.encoder(enc_x, enc_valid_lens, *args)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens, *args)
        return self.decoder(dec_x, dec_state)




