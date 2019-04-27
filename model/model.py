import os

import tensorflow as tf
from config import Config
from data_utils import minibatches, pad_sequence
from tqdm import tqdm_notebook

config = Config()

embedding_size = config.embedding_size


class Model:

    def __init__(self):
        self.embedding_size = config.embedding_size
        self.config = config


    def add_input_op(self):
        with tf.variable_scope('text'):
            self.question_one = tf.placeholder(tf.float32, name='q1', shape=[None, None, self.embedding_size])
            self.question_two = tf.placeholder(tf.float32, name='q2', shape=[None, None, self.embedding_size])

            self.seql_one = tf.placeholder(dtype=tf.int32, name='seql1', shape=[None])
            self.seql_two = tf.placeholder(dtype=tf.int32, name='seql2', shape=[None])
            self.labels = tf.placeholder(dtype=tf.int32, name='labels', shape=[None])


    def get_multirnn_cell(self):
        cells = []
        for _ in range(self.config.n_layers):
            cell = tf.nn.rnn_cell.LSTMCell(self.config.n_hidden, initializer=tf.glorot_uniform_initializer())
            dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                                         input_keep_prob=self.config.keep_prob,
                                                         output_keep_prob=self.config.keep_prob)
            cells.append(dropout_cell)
        return cells


    def add_lstm_op(self):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            cells_fw = self.get_multirnn_cell()
            cells_bw = self.get_multirnn_cell()
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
            (_, _), (state_one_fw, state_one_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                   inputs=self.question_one,
                                                                                   sequence_length=self.seql_one,
                                                                                   dtype=tf.float32)

            self.state_one = tf.concat([state_one_fw[-1].h, state_one_bw[-1].h], name='state_one', axis=-1)

            # self.state_one = tf.concat([state_one_fw, state_one_bw], axis=-1)
            # [batch_size, 2*hidden_size]

            (_, _), (state_two_fw, state_two_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                   inputs=self.question_two,
                                                                                   sequence_length=self.seql_two,
                                                                                   dtype=tf.float32)
            self.state_two = tf.concat([state_two_fw[-1].h, state_two_bw[-1].h], name='state_two', axis=-1)
            # [batch_size, 2*hidden_size]
            # self.state_two = tf.concat([state_two_fw, state_two_bw], axis=-1)


    def add_triplet_loss(self):
        with tf.variable_scope('triplet'):
            self.l2 = tf.norm(self.state_one - self.state_two, 'euclidean', axis=-1)


    def add_logit_op(self):
        if self.config.triplet_loss:

            self.state_one = tf.nn.l2_normalize(self.state_one, axis=-1)
            self.state_two = tf.nn.l2_normalize(self.state_two, axis=-1)

            self.labels = tf.cast(self.labels, tf.float32)
            with tf.variable_scope('triplet'):
                self.l2 = tf.norm(self.state_one -
                                  self.state_two, 'euclidean', axis=-1, name='l2')

        else:
            with tf.variable_scope('proj'):
                self.state = tf.concat([self.state_one, self.state_two], name='state', axis=-1)
                # size[batch_size, 4*hidden_size)

                W = tf.get_variable('W', dtype=tf.float32,
                                    shape=[4 * self.config.n_hidden, self.config.n_tags],
                                    initializer=tf.truncated_normal_initializer())

                b = tf.get_variable('b', shape=[self.config.n_tags],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer())

                self.logits = tf.matmul(self.state, W) + b
                self.pred = tf.nn.softmax(self.logits)
                self.is_duplicate = tf.argmax(self.pred, axis=-1)


    def add_loss_op(self):
        with tf.variable_scope('loss'):
            # self.labels = tf.one_hot(self.labels, depth=self.config.n_tags)

            if self.config.triplet_loss:
                self.loss = tf.reduce_mean(self.labels * self.l2 - (1 - self.labels) * self.l2, name='loss', axis=-1)
                self.pos_l2_mean = tf.reduce_mean(self.labels * self.l2, name='pos_l2')
                self.neg_l2_mean = tf.reduce_mean((1 - self.labels) * self.l2, name='neg_l2')
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels,
                                                                   name='loss'))
            tf.summary.scalar('loss', self.loss)


    def optimize_op(self):
        with tf.variable_scope('opt'):
            global_step = tf.Variable(0, trainable=False)

            starter_learning_rate = self.config.lr
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step,
                self.config.decay_step, self.config.decay_rate, staircase=True
            )

            if self.config.gradient_clipping:
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gvs = optimizer.compute_gradients(self.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                self.optimize = optimizer.apply_gradients(capped_gvs)
            else:
                self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss=self.loss,
                                                                           global_step=global_step,
                                                                           # var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=[])
                                                                           )
            if not self.config.triplet_loss:
                self.mistakes = tf.equal(tf.argmax(self.pred, axis=-1, output_type=tf.int32), self.labels)

                self.accuracy = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))


    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def save_session(self):
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        self.saver.save(self.sess, self.config.save_dir)


    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.saver.restore(self.sess, dir_model)


    def add_summary(self):
        """
        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.summary_dir,
                                                 self.sess.graph)


    def run_epoch(self, train_data, dev_data, epoch):
        pad_item = [0] * Config.embedding_size
        n_batches = int((len(train_data) + self.config.batch_size - 1) / self.config.batch_size)

        t_loss = 0
        c = 0
        tot_ac = 0
        pos_loss = 0
        neg_loss = 0
        for i, (sent_1, sent_2, sent_1_lengths, sent_2_lengths, label) in tqdm_notebook(enumerate(
                minibatches(train_data,
                            self.config.batch_size))):
            max_len_1 = max(sent_1_lengths)
            max_len_2 = max(sent_2_lengths)

            sent_1, _ = pad_sequence(sent_1, max_len_1, pad_item)
            sent_2, _ = pad_sequence(sent_2, max_len_2, pad_item)
            feed_dict = self.get_feed_dict(
                sent_1,
                sent_2,
                sent_1_lengths,
                sent_2_lengths,
                label
            )
            reverse_feed_dict = self.get_feed_dict(sent_2,
                                                   sent_1,
                                                   sent_2_lengths,
                                                   sent_1_lengths,
                                                   label)
            try:
                if self.config.triplet_loss:
                    _, loss, summary, pos_l2, neg_l2 = self.sess.run([self.optimize,
                                                                      self.loss,
                                                                      self.merged,
                                                                      self.pos_l2_mean,
                                                                      self.neg_l2_mean], feed_dict=feed_dict)
                    pos_loss += pos_l2
                    neg_loss += neg_l2
                else:

                    _, loss, summary, accuracy = self.sess.run([
                        self.optimize, self.loss, self.merged, self.accuracy], feed_dict=feed_dict
                    )
                    _, loss, summary = self.sess.run([
                        self.optimize, self.loss, self.merged], feed_dict=reverse_feed_dict
                    )
                    tot_ac += accuracy

                if i % self.config.summary_freq:
                    self.file_writer.add_summary(summary=summary)

                t_loss += loss
                c += 1
            except Exception as e:
                print(str(e))
        print('At epoch {} loss is..{}'.format(epoch, str(float(t_loss) / n_batches)))

        if self.config.triplet_loss:
            print(('At epoch {} training pos l2 mean is..{}'.format(epoch, str(float(pos_loss) / c))))
            print(('At epoch {} training neg l2 mean is..{}'.format(epoch, str(float(neg_loss) / c))))
        else:
            print('At epoch {} training accuracy is..{}'.format(epoch, str(float(tot_ac) / c)))

        c = 0
        tot_ac = 0
        pos_loss = 0
        neg_loss = 0
        for i, (dev_texts_1, dev_texts_2, dev_texts_lens_1, dev_texts_lens_2, dev_label) in tqdm_notebook(enumerate(
                minibatches(dev_data, self.config.batch_size))):
            max_len_1 = max(dev_texts_lens_1)
            max_len_2 = max(dev_texts_lens_2)

            dev_texts_1, _ = pad_sequence(dev_texts_1, max_len_1, pad_item)
            dev_texts_2, _ = pad_sequence(dev_texts_2, max_len_2, pad_item)
            try:
                if self.config.triplet_loss:
                    pos_l2, neg_l2 = self.sess.run(
                        [self.pos_l2_mean, self.neg_l2_mean], feed_dict=self.get_feed_dict(
                            dev_texts_1,
                            dev_texts_2,
                            dev_texts_lens_1,
                            dev_texts_lens_2,
                            dev_label
                        )
                    )
                    pos_loss += pos_l2
                    neg_loss += neg_l2
                else:
                    accr = self.sess.run(
                        self.accuracy, feed_dict=self.get_feed_dict(
                            dev_texts_1,
                            dev_texts_2,
                            dev_texts_lens_1,
                            dev_texts_lens_2,
                            dev_label
                        )
                    )
                    tot_ac += accr

                c += 1
            except Exception as e:
                print(str(e))

        if self.config.triplet_loss:
            print(('At epoch {} test pos l2 mean is..{}'.format(epoch, str(float(pos_loss) / c))))
            print(('At epoch {} test neg l2 mean is..{}'.format(epoch, str(float(neg_loss) / c))))
        else:
            print('At epoch {} dev acc..{}'.format(epoch, str(float(tot_ac) / c)))
        print('=' * 50)
        # break


    def train(self, train_data, dev_data):
        self.add_summary()

        for i in range(self.config.epoch):
            self.run_epoch(train_data, dev_data, i)

            if i % self.config.saving_freq == 0:
                self.save_session()


    def get_feed_dict(self,
                      sent_1,
                      sent_2,
                      sent_1_len,
                      sent_2_len,
                      label=None
                      ):
        feed_data = {
            self.question_one: sent_1,
            self.question_two: sent_2,
            self.seql_one: sent_1_len,
            self.seql_two: sent_2_len,

        }
        if label:
            feed_data[self.labels] = label

        return feed_data


    def build(self):
        self.add_input_op()
        self.add_lstm_op()
        self.add_logit_op()
        self.add_loss_op()
        self.optimize_op()
        self.initialize_session()
