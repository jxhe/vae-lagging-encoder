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

    def decode_sentence(self, sentence):
        decoded_sentence = []
        for wid_t in sentence:
            wid = wid_t.item()
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence


    @staticmethod
    def from_corpus(fname):
        vocab = VocabEntry()
        with open(fname) as fin:
            for line in fin:
                _ = [vocab.add(word) for word in line.split()]

        return vocab


class MonoTextData(object):
    """docstring for MonoTextData"""
    def __init__(self, fname, label=False, max_length=None, vocab=None):
        super(MonoTextData, self).__init__()

        self.data, self.vocab, self.dropped, self.labels = self._read_corpus(fname, label, max_length, vocab)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, label, max_length, vocab):
        data = []
        labels = [] if label else None
        dropped = 0
        if not vocab:
            vocab = defaultdict(lambda: len(vocab))
            vocab['<pad>'] = 0
            vocab['<s>'] = 1
            vocab['</s>'] = 2
            vocab['<unk>'] = 3

        with open(fname) as fin:
            for line in fin:
                if label:
                    split_line = line.split('\t')
                    lb = split_line[0]
                    split_line = split_line[1].split()
                else:
                    split_line = line.split()
                if len(split_line) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(lb)
                data.append([vocab[word] for word in split_line])

        if isinstance(vocab, VocabEntry):
            return data, vocab, dropped, labels

        return data, VocabEntry(vocab), dropped, labels

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
        Returns:
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
            batch_data.sort(key=lambda e: -len(e))

            batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)

            yield batch_data, sents_len

    def create_data_batch_labels(self, batch_size, device, batch_first=False):
        """pad data with start and stop symbol, batching is performerd w.r.t.
        the sentence length, so that each returned batch has the same length,
        no further pack sequence function (e.g. pad_packed_sequence) is required
        Returns: List
            List: a list of batched data, each element is a tensor with shape
                (seq_len, batch_size)
        """
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        # record the locations where length changes
        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i-1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        total = 0
        curr = 0
        for idx in change_loc:
            while curr < idx:
                batch_data = []
                batch_label = []
                next = min(curr + batch_size, idx)
                for id_ in range(curr, next):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])
                curr = next
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert(sents_len == ([sents_len[0]] * len(sents_len)))

        assert(total == len(self.data))
        return batch_data_list, batch_label_list

    def create_data_batch(self, batch_size, device, batch_first=False):
        """pad data with start and stop symbol, batching is performerd w.r.t.
        the sentence length, so that each returned batch has the same length,
        no further pack sequence function (e.g. pad_packed_sequence) is required
        Returns: List
            List: a list of batched data, each element is a tensor with shape
                (seq_len, batch_size)
        """
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        # record the locations where length changes
        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i-1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        total = 0
        curr = 0
        for idx in change_loc:
            while curr < idx:
                batch_data = []
                next = min(curr + batch_size, idx)
                for id_ in range(curr, next):
                    batch_data.append(self.data[sort_idx[id_]])
                curr = next
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)

                total += batch_data.size(0)
                assert(sents_len == ([sents_len[0]] * len(sents_len)))

        assert(total == len(self.data))
        return batch_data_list


    def data_sample(self, nsample, device, batch_first=False, shuffle=True):
        """sample a subset of data (like data_iter)
        Returns:
            batch_data: LongTensor with shape (seq_len, batch_size)
            sents_len: list of data length, this is the data length
                       after counting start and stop symbols
        """

        index_arr = np.arange(len(self.data))

        if shuffle:
            np.random.shuffle(index_arr)

        batch_ids = index_arr[: nsample]
        batch_data = [self.data[index] for index in batch_ids]

        # uncomment this line if the dataset has variable length
        batch_data.sort(key=lambda e: -len(e))

        batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)

        return batch_data, sents_len
