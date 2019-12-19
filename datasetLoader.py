from torchtext import data, datasets, vocab
import spacy
from torchtext.vocab import Vectors


class Dataset:
    def __init__(self, dataset="wmt14"):
        self.BOS_WORD = '<s>'
        self.EOS_WORD = '</s>'
        self.BLANK_WORD = "<blank>"
        self.MAX_LEN = 100
        self.MIN_FREQ = 2
        if dataset == "iwslt":
            self.train, self.val, self.test, self.src_vocab, self.tgt_vocab = self.IWSLT()
        elif dataset == "wmt14":
            self.train, self.val, self.test, self.src_vocab, self.tgt_vocab = self.WMT14()

    def IWSLT(self):
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = data.Field(tokenize=tokenize_de, pad_token=self.BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token=self.BOS_WORD,
                         eos_token=self.EOS_WORD, pad_token=self.BLANK_WORD)

        train, val, test = datasets.IWSLT.splits(
                            exts=('.de', '.en'), fields=(SRC, TGT),
                            filter_pred=lambda x: len(vars(x)['src']) <= self.MAX_LEN and
                                                    len(vars(x)['trg']) <= self.MAX_LEN)

        SRC.build_vocab(train.src, min_freq=self.MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=self.MIN_FREQ)
        return train, val, test, SRC.vocab, TGT.vocab

    def WMT14(self):
        def tokenize(text):
            return text.split(" ")

        print("test1")
        SRC = data.Field(tokenize=tokenize, pad_token=self.BLANK_WORD)
        TGT = data.Field(tokenize=tokenize, init_token=self.BOS_WORD,
                         eos_token=self.EOS_WORD, pad_token=self.BLANK_WORD)
        print("test2")
        def tk(text):
            print(text)
            return text.replace("\n", "")

        train, val, test = datasets.WMT14.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= self.MAX_LEN and
                                  len(vars(x)['trg']) <= self.MAX_LEN)
        print("test")
        TGT.build_vocab(train.src, train.trg, min_freq=self.MIN_FREQ)
        VOCAB = TGT.vocab
        return train, val, test, VOCAB, VOCAB


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
