import torch
from torch.autograd import Variable
from datasetHandler import *
from datasetLoader import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# load model
from training import run_epoch, SimpleLossCompute, LabelSmoothing, NoamOpt


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    batch = src.size(0)
    memory = model.encode(src, src_mask)
    ys = torch.ones(batch, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
    return ys


def translation(out, trg):
    assert out.size(0) == trg.size(0)
    for x in range(0, out.size(0)):
        print("Translation:", end="\t")
        for j in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[x, j]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for j in range(1, trg.size(1)):
            sym = TGT.vocab.itos[trg.data[x, j]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()


def bleu(outs, trgs):
    bleu_value1 = 0
    bleu_value2 = 0
    bleu_value3 = 0
    bleu_value4 = 0
    num = 0
    for out, trg in zip(outs, trgs):
        assert out.size(0) == trg.size(0)
        for x in range(0, out.size(0)):
            num = num + 1
            candidate = []
            for j in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[x, j]]
                if sym == "</s>": break
                candidate.append(sym)
            tmp = []
            for j in range(1, trg.size(1)):
                sym = TGT.vocab.itos[trg.data[x, j]]
                if sym == "</s>": break
                tmp.append(sym)
            reference = [tmp]
            tmp1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            tmp2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            tmp3 = sentence_bleu(reference, candidate, weights=(0.33333, 0.33333, 0.33333, 0))
            tmp4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            bleu_value1 = bleu_value1 + tmp1
            bleu_value2 = bleu_value2 + tmp2
            bleu_value3 = bleu_value3 + tmp3
            bleu_value4 = bleu_value4 + tmp4
    return bleu_value1 / num, bleu_value2 / num, bleu_value3 / num, bleu_value4 / num


if __name__ == "__main__":
    model = torch.load("./models_4/195.pkl")
    model.cuda()
    model.eval()
    BATCH_SIZE = 1200
    pad_idx = TGT.vocab.stoi["<blank>"]
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device(0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=torch.device(0),
                           repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=False)
    outs = []
    trgs = []
    for i, batch in enumerate((rebatch(pad_idx, b) for b in test_iter)):
        # print(i, batch.src)
        src = batch.src
        src_mask = batch.src_mask
        trg = batch.trg

        out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        translation(out, trg)
        outs.append(out)
        trgs.append(trg)
        # break

    print(bleu(outs, trgs))