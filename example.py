import torch
from torch.autograd import Variable
import numpy as np
from training import Batch, LabelSmoothing, NoamOpt, run_epoch
from model import make_model, subsequent_mask


def data_gen(V, batch, nbatches):
	""
	for i in range(nbatches):
		data = torch.from_numpy(np.random.randint(1, V, size=(batch,10)).astype(np.int64))
		data[:,0] = 1
		src = Variable(data, requires_grad=False)
		tgt = Variable(data, requires_grad=False)
		yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))