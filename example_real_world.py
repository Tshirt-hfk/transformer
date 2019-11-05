# For data loading.
import torch
from torch import nn
from torch.autograd import Variable
from torchtext import data, datasets

from example import SimpleLossCompute
from model import make_model, subsequent_mask, make_model_1, make_model_2

from training import LabelSmoothing, batch_size_fn, Batch, NoamOpt, run_epoch

if True:
    import spacy

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')


    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]


    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]


    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    if False:
        model = torch.load("./models_1/239.pkl")
    else:
        model = make_model_2(len(SRC.vocab), len(TGT.vocab), N=4, h=8)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 600
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device(0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device(0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)



if True:
    # train model
    print("start training model!")
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(1, 300):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                         SimpleLossCompute(model.generator, criterion, model_opt))
        torch.save(model, "./models_2/" + str(epoch) + ".pkl")
        with open("./models_2/data.txt", "a+") as f:
            f.write(str(epoch) + ":" + str(loss) + "\n")
        print(epoch, loss)

else:
    # load model
    model = torch.load("iwslt.pt")


def greedy_decode(model, src, src_mask, src_mask_p, max_len, start_symbol):
    memory = model.encode(src, src_mask, src_mask_p)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


for i, batch in enumerate((rebatch(pad_idx, b) for b in valid_iter)):
    src = batch.src
    src_mask = batch.src_mask
    src_mask_p = batch.src_mask_p

    out = greedy_decode(model, src, src_mask, src_mask_p,
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for j in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, j]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()
    print("Target:", end="\t")
    for j in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[j, 0]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()
    break
