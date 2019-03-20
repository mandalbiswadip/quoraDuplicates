import numpy as np
import pandas as pd

from embedding import get_embedding

class Data:


    def __init__(self, filename, batch_size = None):
        if batch_size:
            self.batch_size = batch_size
        self.dataset = pd.read_csv(filename)

    def __iter__(self):
        q1, q2, seql1, seql2, y = [], [], [], [], []
        c = 0
        for i, row in enumerate(self.dataset):
            if c == self.batch_size or i == len(self.dataset):
                yield q1, q2, seql1, seql2, y
                q1, q2, seql1, seql2, y = [], [], [], [], []
            if i == len(self.dataset):
                break

            q1.append( np.array([get_embedding(w) for w in row['question1'].split()]))
            q2.append( np.array([get_embedding(w) for w in row['question2'].split()]))
            seql1 += [len(row['question1'])]
            seql2 += [len(row['question2'])]
            y += [row['is_duplicate']]
            c+=1






def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch, z_batch, a_batch, b_batch = [], [], [], [], []
    for (x, y, z, a, b) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch, a_batch, b_batch
            x_batch, y_batch, z_batch, a_batch, b_batch = [], [], [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        z_batch += [z]
        a_batch += [a]
        b_batch += [b]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch, a_batch, b_batch

def pad_sequence(seqs, max_len, pad_by):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with

        Returns:
            a list of list where each sublist has same length
        """
    sequence_padded, sequence_length = [], []

    for seq in seqs:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_by] * max(max_len - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_len)]

    return sequence_padded, sequence_length
