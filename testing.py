import torch
from torch.autograd import Variable
from datasetHandler import subsequent_mask


# load model
# model = torch.load("./models_4/105.pkl")


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
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

# for i, batch in enumerate((rebatch(pad_idx, b) for b in valid_iter)):
#     src = batch.src
#     src_mask = batch.src_mask
#     src_mask_p = batch.src_mask_p
#
#     out = greedy_decode(model, src, src_mask, src_mask_p,
#                         max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
#     print("Translation:", end="\t")
#     for j in range(1, out.size(1)):
#         sym = TGT.vocab.itos[out[0, j]]
#         if sym == "</s>": break
#         print(sym, end=" ")
#     print()
#     print("Target:", end="\t")
#     for j in range(1, batch.trg.size(0)):
#         sym = TGT.vocab.itos[batch.trg.data[j, 0]]
#         if sym == "</s>": break
#         print(sym, end=" ")
#     print()
#     break
