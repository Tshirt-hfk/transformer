import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # scores (b, h, l, l)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(scores.size(), mask.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.d_model = d_model
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.choices = clones(nn.Linear(self.d_k, d_model), h)

    def forward(self, query, key, value, mask=None, mask_p=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # x = x.transpose(1, 2).contiguous().view(
        #     nbatches, -1, self.h * self.d_k)
        #
        # return self.linears[-1](x)
        # x (b, h, l, d_k)
        x = x.split(1, 1)
        x = [l(a).view(nbatches, -1, self.d_model) for l, a in zip(self.choices, x)]
        x = F.relu(torch.stack(x, dim=-2)).sum(-2)
        return x

class MyMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MyMultiHeadedAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.linears = clones(nn.Linear(d_model, d_model * h), 3)
        self.linears_p = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.attn_p = None
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, mask_p=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        if mask_p is not None:
            mask_p = mask_p.unsqueeze(1)

        nbatches = query.size(0)

        query_x, key_x, value_x = \
            [l(x).view(nbatches, -1, self.h, self.d_model).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # x (b, h, v, d_model)
        x, self.attn = attention(query_x, key_x, value_x, mask=mask,
                                 dropout=self.dropout)
        # predict
        query_p, key_p, value_p = \
            [l(x).view(nbatches, -1, 1, self.d_model).transpose(1, 2)
             for l, x in zip(self.linears_p, (query, key, value))]
        # mask_p = torch.from_numpy(np.eye(query_p.size(2), query_p.size(2), dtype="uint8")) == 0

        # x (b, 1, h, d_model)
        x_p, self.attn_p = attention(query_p, key_p, value_p, mask=mask_p,
                                     dropout=self.dropout)
        # compute similarity
        # dis (b, l, h)
        dis = torch.sum((x - x_p.expand(-1, self.h, -1, -1)) ** 2, dim=3).unsqueeze(3).transpose(1, 2)
        output = torch.matmul(F.softmax(1 / (dis + 1e-9), dim=2).transpose(-1, -2), x.transpose(1, 2)).squeeze(-2)
        # print(output.size(), x_p.size())
        return self.linear(output * 0.8 + x_p.squeeze(1) * 0.2)


class MultiWayAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiWayAttention, self).__init__()
        self.wayNum = 2
        self.d_model = d_model
        self.ways = clones(MultiHeadedAttention(h, d_model, dropout), self.wayNum)
        self.pred = MultiHeadedAttention(h, d_model, dropout)
        self.keyW = nn.Linear(d_model, d_model)
        self.queryW = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, mask_p=None):
        nbatches = query.size(0)
        # x_p (b, l, d_model)
        x_p = self.pred(query, key, value, mask_p, None)
        # xs (b, l, w, d_model)
        xs = torch.stack([l(query, key, value, mask, None) for l in self.ways], 2)
        # print(xs.size())
        # query (b, l, d_model)
        query = self.queryW(x_p)
        # key (b, l, w, d_model)
        key = self.keyW(xs)
        # print(query.size(), key.size())
        # scores (b, l, w)
        scores = torch.matmul(query.unsqueeze(-2), key.transpose(-2, -1)).squeeze(-2)
        index = torch.max(scores, dim=-1)[1].unsqueeze(-1).long()
        # print(index.size())
        mask_output = torch.zeros((index.size(0), index.size(1), self.wayNum)).cuda().scatter(-1, index, 1)
        # print(mask_output.size())
        output = torch.sum(xs.masked_fill(mask_output.unsqueeze(-1).expand(-1, -1, -1, self.d_model) == 0, 0), dim=-2).squeeze(-2)
        # print(output.size())
        return output


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


# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe.forward(Variable(torch.ones(1, 100, 20)))
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])
# plt.show()


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

    def forward(self, x, mask, mask_p):
        for layer in self.layers:
            x = layer(x, mask, mask_p)
        return self.norm(x)


class EncoderLayer(nn.Module):
    ""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.self_attn = self_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, mask_p):
        ""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, mask_p))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mak):
        ""
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

    def forward(self, src, tgt, src_mask, src_mask_p, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask, src_mask_p), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask, src_mask_p):
        return self.encoder(self.src_embed(src), src_mask, src_mask_p)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # attn = MyMultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def make_model_1(src_vocab, tgt_vocab, N=6,
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # attn = MyMultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def make_model_2(src_vocab, tgt_vocab, N=6,
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn1 = MultiWayAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn1), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == "__main__":
    tmp_model = make_model(10, 10, 2)
    print(tmp_model)
