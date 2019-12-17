import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable

from datasetLoader import *
from datasetHandler import *
from model import make_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_epoch(data_iter, model, loss_compute):
    """"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        tokens += batch.ntokens
        total_tokens += batch.ntokens
        if i % 100 == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / torch.tensor(elapsed)))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, model, generator, criterion, opt=None, train=True):
        self.model = model
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.train = train
        self.reg = 1e-6

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # l1_loss = Variable(torch.FloatTensor([0]), requires_grad=True).cuda()
        # for name, param in self.model.named_parameters():
        #     if "choices_parm" in name:
        #         l1_loss = l1_loss + (self.reg * torch.sum(torch.abs(param)))
        # print(loss, l1_loss)
        # loss1 = l1_loss + loss
        if self.train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        return loss.data * norm


def start_train():
    pad_idx = TGT.vocab.stoi["<blank>"]
    if False:
        print("loading model!")
        model = torch.load("./models/205.pkl")
    else:
        model = make_model(len(SRC.vocab), len(TGT.vocab), t=0, N=4, h=8, d_k=64)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 1000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device(0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device(0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=torch.device(0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # train model
    print("start training model!")
    for epoch in range(1, 500):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                  SimpleLossCompute(model, model.generator, criterion, model_opt, True))
        model.eval()
        loss1 = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                         SimpleLossCompute(model, model.generator, criterion, None, False))
        loss2 = run_epoch((rebatch(pad_idx, b) for b in test_iter), model,
                         SimpleLossCompute(model, model.generator, criterion, None, False))
        torch.save(model, "./models_2/" + str(epoch) + ".pkl")
        with open("./models_2/data.txt", "a+") as f:
            f.write(str(epoch) + ":" + str(loss1) + "\t" + str(loss2) + "\n")
        print(epoch, loss)


if __name__ == "__main__":
    start_train()
