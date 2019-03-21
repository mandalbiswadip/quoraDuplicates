import re



def tokenize_sent(sent):
    tokens = []
    if sent:
        tokens = [x for x in re.split(r'\W+',sent) if x]
    return tokens

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
