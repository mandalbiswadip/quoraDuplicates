import warnings

warnings.filterwarnings("ignore")

from model import Model
from config import Config
from embedding import get_embedding
from data_utils import tokenize_sent

config = Config()

dup_dict = {
    0: 'No',
    1: 'Yes'
}

model = Model()
model.build()
model.restore_session(config.save_dir)


def get_if_duplicate(sentence1, sentence2):
    is_duplicate = 0
    try:
        global model
        if sentence1 and sentence2:
            sentence1 = tokenize_sent(str(sentence1).lower())
            sentence2 = tokenize_sent(str(sentence2).lower())
            sentence1 = [get_embedding(w) for w in sentence1]
            sentence2 = [get_embedding(w) for w in sentence2]

            len1 = len(sentence1)
            len2 = len(sentence2)

            results, pred = model.sess.run(
                [model.is_duplicate, model.pred],
                feed_dict=model.get_feed_dict(
                    [sentence1],
                    [sentence2],
                    [len1],
                    [len2],
                    [0]
                )
            )
            is_duplicate = results[0]
            print(pred[0])

    except Exception as e:
        print(str(e))
    return dup_dict[is_duplicate]


if __name__ == "__main__":
    import sys

    sentence1 = sys.argv[1]
    sentence2 = sys.argv[2]

    print(get_if_duplicate(sentence1, sentence2))
