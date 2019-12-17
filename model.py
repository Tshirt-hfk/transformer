import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # scores (b, h, l, l)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # print(p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_k, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.d_model = d_model
        self.h = h
        self.linears = clones(nn.Linear(d_model, h*d_k), 3)
        self.linears_last = nn.Linear(h*d_k, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears_last(x)


class MultiHeadedAttention1(nn.Module):
    def __init__(self, h, d_k, d_model, dropout=0.1):
        super(MultiHeadedAttention1, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.d_model = d_model
        self.h = h
        self.linears = clones(nn.Linear(d_model, h*d_k), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.choices = clones(nn.Linear(self.d_k, d_model), h)
        # self.choices_parm = nn.Parameter(torch.ones(self.h, self.d_model))
        # self.choices_bias = nn.Parameter(torch.zeros(self.d_model))
        # self.choices = nn.Parameter(torch.Tensor(1, 1, self.h, self.d_model, self.d_k))
        # self.choices_bias = nn.Parameter(torch.Tensor(h, self.d_model))

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # x (b, h, l, d_k)
        x = x.split(1, 1)
        x = [l(a).view(nbatches, -1, self.d_model) for l, a in zip(self.choices, x)]
        x = torch.stack(x, dim=-1)
        x0 = x.abs()
        x1 = x0.max(dim=-1)[0]
        x2 = (x0/(x1.unsqueeze(-1)+1e-2)*x).sum(dim=-1)
        # print(x1.size(), x2.size())
        return x2

        # x = x.view(nbatches, -1, self.h, self.d_k, 1)
        # print(x.size())
        # c = self.choices.expand(nbatches, x.size(1), -1, -1, -1)
        # print(c.size(), type(c))
        # x = torch.matmul(c, x)\
        #     .view(nbatches, -1, self.h, self.d_model)
        # x = (x+self.choices_bias).max(dim=-2)[0]
        # return x


class MultiHeadedAttention2(nn.Module):
    def __init__(self, h, d_k, d_model, dropout=0.1):
        super(MultiHeadedAttention2, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.d_model = d_model
        self.h = h
        self.linears = clones(nn.Linear(d_model, h*d_k), 3)
        self.linears_last = nn.Linear(h*d_k, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.midLinears = clones(nn.Linear(self.d_k, self.d_k), h)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # x (b, h, l, d_k)
        x = x.split(1, 1)
        x = [l(a) for l, a in zip(self.midLinears, x)]
        x = torch.stack(x, dim=-2).view(nbatches, -1, self.h * self.d_k)
        return self.linears_last(x)


class MultiHeadedAttention3(nn.Module):
    def __init__(self, h, d_k, d_model, dropout=0.1):
        super(MultiHeadedAttention1, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.d_model = d_model
        self.h = h
        self.linears = clones(nn.Linear(d_model, h*d_k), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.choices = clones(nn.Linear(self.d_k, d_model), h)
        # self.choices = nn.Parameter(torch.Tensor(1, 1, self.h, self.d_model, self.d_k))
        # self.choices_bias = nn.Parameter(torch.Tensor(h, self.d_model))

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # x (b, h, l, d_k)
        x = x.split(1, 1)
        x = [l(a).view(nbatches, -1, self.d_model) for l, a in zip(self.choices, x)]
        x = F.celu(torch.stack(x, dim=-1)).sum(dim=-1)
        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    ""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.self_attn = self_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        ys = []
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            ys.append(x)
        ys = torch.stack(ys, dim=-2)
        return ys


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mak):
        """"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mak))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class EncoderDecoder(nn.Module):
    """
	A standard Encoder-Decoder architecture. Base for this and many 
    other models.
	"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask,), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, N):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        k = [math.log(i+1, 2) for i in range(N)]
        k = torch.Tensor(k).cuda()
        self.k = Variable((k / k.sum()).unsqueeze(-1).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x = self.proj(x)
        x = (self.k*x).sum(dim=-2)
        return F.log_softmax(x, dim=-1)


def make_model(src_vocab, tgt_vocab, t=0, N=6,
               d_k=64, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    if t == 0:
        attn = MultiHeadedAttention(h, d_k, d_model)
    elif t == 1:
        attn = MultiHeadedAttention1(h, d_k, d_model)
    elif t == 2:
        attn = MultiHeadedAttention2(h, d_k, d_model)
    else:
        attn = MultiHeadedAttention3(h, d_k, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab, N)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == "__main__":
    tmp_model = make_model(10, 10, 0, 2)
    print(tmp_model)
