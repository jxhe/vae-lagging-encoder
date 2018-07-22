import random
import torch
import numpy as np

from collections import defaultdict


class VocabEntry(object):
    """docstring for Vocab"""
    def __init__(self, word2id=None):
        super(VocabEntry, self).__init__()

        if word2id:
            self.word2id = word2id
            self.unk_id = word2id['<unk>']
        else:
            self.word2id = dict()
            self.unk_id = 3
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = self.unk_id

        self.id2word_ = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid

        else:
            return self[word]

    def id2word(self, wid):
        return self.id2word_[wid]


    @staticmethod
    def from_corpus(fname):
        vocab = VocabEntry()
        with open(fname) as fin:
            for line in fin:
                _ = [vocab.add(word) for word in line.split()]

        return vocab


class MonoTextData(object):
    """docstring for MonoTextData"""
    def __init__(self, fname, vocab=None):
        super(MonoTextData, self).__init__()

        self.data, self.vocab = self._read_corpus(fname, vocab)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, vocab):
        data = []
        if not vocab:
            vocab = defaultdict(lambda: len(vocab))
            vocab['<pad>'] = 0
            vocab['<s>'] = 1
            vocab['</s>'] = 2
            vocab['<unk>'] = 3

        with open(fname) as fin:
            for line in fin:
                data.append([vocab[word] for word in line.split()])

        if isinstance(vocab, VocabEntry):
            return data, vocab

        return data, VocabEntry(vocab)

    def _to_tensor(self, batch_data, batch_first, device):
        """pad a list of sequences, and transform them to tensors
        Args:
            batch_data: a batch of sentences (list) that are composed of
                word ids. 
            batch_first: If true, the returned tensor shape is 
                (batch, seq_len), otherwise (seq_len, batch)
            device: torch.device

        Returns: Tensor, Int list
            Tensor: Tensor of the batch data after padding
            Int list: a list of integers representing the length
                of each sentence (including start and stop symbols)
        """


        # pad stop symbol
        batch_data = [sent + [self.vocab['</s>']] for sent in batch_data]

        sents_len = [len(sent) for sent in batch_data]

        max_len = max(sents_len)

        batch_size = len(sents_len)
        sents_new = []

        # pad start symbol
        sents_new.append([self.vocab['<s>']] * batch_size)
        for i in range(max_len):
            sents_new.append([sent[i] if len(sent) > i else self.vocab['<pad>'] \
                               for sent in batch_data])


        sents_ts = torch.tensor(sents_new, dtype=torch.long, 
                                 requires_grad=False, device=device)

        if batch_first:
            sents_ts = sents_ts.permute(1, 0).contiguous()

        return sents_ts, [length + 1 for length in sents_len]


    def data_iter(self, batch_size, device, batch_first=False, shuffle=True):
        """pad data with start and stop symbol, and pad to the same length
        Return:
            batch_data: LongTensor with shape (seq_len, batch_size)
            sents_len: list of data length, this is the data length
                       after counting start and stop symbols
        """
        index_arr = np.arange(len(self.data))

        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(index_arr)) / float(batch_size))
        for i in range(batch_num):
            batch_ids = index_arr[i * batch_size : (i+1) * batch_size]
            batch_data = [self.data[index] for index in batch_ids]

            # uncomment this line if the dataset has variable length
            # batch_data.sort(key=lambda e: -len(e))

            batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)

            yield batch_data, sents_len