import os

path = os.path.realpath(__file__)

PROJECT_HOME = '/'.join(path.split('/')[:-2])


class Config:

    def __init__(self):
        pass

    n_hidden = 256
    n_layers = 2
    embedding_size = 300
    n_tags = 2
    # max_sent_len = 70

    #learning
    lr = 0.01
    decay_step = 1000
    decay_rate = 0.9
    dropout_prob = 0.5

    epoch = 100
    batch_size = 256
    saving_freq = 2
    summary_freq = 10

    save_dir = PROJECT_HOME + '/models_weights/'
    summary_dir = PROJECT_HOME + '/results/'
    embedding_path = PROJECT_HOME + '/GoogleNews-vectors-negative300.bin'