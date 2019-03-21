import warnings
warnings.filterwarnings("ignore")

from flask import Flask
from flask_cors import CORS

import tensorflow as tf

from model import Model
from config import Config
from embedding import get_embedding
from data_utils import tokenize_sent

config = Config()

app = Flask(__name__)

CORS(app)

dup_dict = {
    0:'No',
    1:'Yes'
}

model = Model()
model.build()
model.restore_session(config.save_dir)


@app.route('/sentence1=<sentence1>&sentence2=<sentence2>', methods=['GET', 'POST'])
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


            results = model.sess.run(
                        model.is_duplicate,
                        feed_dict = model.get_feed_dict(
                                        [sentence1],
                                        [sentence2],
                                        [len1],
                                        [len2],
                                        None
                    )
            )
            is_duplicate = results[0]

    except Exception as e:
        print(str(e))
    return dup_dict[is_duplicate]

if __name__=="__main__":
    app.debug = True
    app.run(host='0.0.0.0', threaded=True)
