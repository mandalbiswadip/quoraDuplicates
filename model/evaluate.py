import warnings

warnings.filterwarnings("ignore")

from model import Model
from config import Config
from embedding import get_embedding

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
            sentence1 = str(sentence1).lower().split()
            sentence2 = str(sentence2).lower().split()
            sentence1 = [get_embedding(w) for w in sentence1]
            sentence2 = [get_embedding(w) for w in sentence2]

            len1 = len(sentence1)
            len2 = len(sentence2)

            is_duplicate = model.sess.run(
                model.is_duplicate,
                feed_dict=model.get_feed_dict(
                    sentence1,
                    sentence2,
                    len1,
                    len2,
                    None
                )
            )

    except Exception as e:
        pass
    return dup_dict[is_duplicate]


if __name__ == "__main__":
    import sys

    sentence1 = sys.argv[1]
    sentence2 = sys.argv[2]

    print(get_if_duplicate(sentence1, sentence2))
