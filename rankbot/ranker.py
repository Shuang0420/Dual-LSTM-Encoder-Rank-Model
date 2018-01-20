# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

"""
Model to predict the matching score between query and responses.

"""
import numpy as np
import tensorflow as tf
import os

class Ranker:
    """ A retrieval-based chatbot.
    Architecture: LSTM Encoder/Encoder.
    """

    def __init__(self, args, textData, mode="train", sess=None):
        """
        Args:
            args: parameters of the model。
            textData: the dataset object
            mode: ["train", "valid", "eval", "predict", "cache"]
            sess: the current session
        """
        print("Model creation...")

        self.mode = mode


        self.args = args
        self.textData = textData  # Keep a reference on the dataset
        self.dtype = tf.float32

        self.optOp = None
        self.outputs = None

        # for predicting
        self.ranksize = 0
        self.sess = sess
        self.modelDir = os.path.join(args.rootDir, 'save', 'model')


        if args.modelTag:
            print("modelTag=%s" % args.modelTag)
            self.modelDir += '-' + args.modelTag

        self.cachePath = os.path.join(self.modelDir, 'cache')
        self.bestModelPath = os.path.join(self.modelDir, 'best_model.ckpt')

        # Construct graph
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(
                self.args.hiddenSize,
            )
            if self.mode=="train":
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_keep_prob=self.args.dropout
                )
            return cell

        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)
        with tf.name_scope('placeholder_query'):
            self.query_seqs  = tf.placeholder(tf.int32, [None, None], name='query')
            self.query_length  = tf.placeholder(tf.int32, [None], name='query_length')
        with tf.name_scope('placeholder_labels'):
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
            self.targets = tf.placeholder(tf.int32, [None], name='targets')


        with tf.name_scope('placeholder_response'):
            self.response_seqs = tf.placeholder(tf.int32, [None, None], name='response')
            self.response_length = tf.placeholder(tf.int32, [None], name='response_length')



        with tf.name_scope('embedding_layer'):
            self.embedding = tf.get_variable('embedding',
                                             [len(self.textData.word2id), self.args.embeddingSize])
            self.embed_query = tf.nn.embedding_lookup(self.embedding, self.query_seqs)
            self.embed_response = tf.nn.embedding_lookup(self.embedding, self.response_seqs)
            if self.mode=="train" and self.args.dropout > 0:
                self.embed_query = tf.nn.dropout(self.embed_query, keep_prob = self.args.dropout)
                self.embed_response = tf.nn.dropout(self.embed_response, keep_prob = self.args.dropout)


        query_output, query_final_state = tf.nn.dynamic_rnn(
            cell = encoder_cell,
            inputs = self.embed_query,
            sequence_length = self.query_length,
            time_major = False,
            dtype=tf.float32)


        response_output, response_final_state = tf.nn.dynamic_rnn(
            cell = encoder_cell,
            inputs = self.embed_response,
            sequence_length = self.response_length,
            time_major = False,
            dtype=tf.float32)


        with tf.variable_scope('bilinar_regression'):
            W = tf.get_variable("bilinear_W",
                                shape=[self.args.hiddenSize, self.args.hiddenSize],
                                initializer=tf.truncated_normal_initializer())


        if self.mode=="train":
            # Use other responses in minibatch as negative response
            response_final_state = tf.matmul(response_final_state[-1].h, W)
            logits = tf.matmul(
                a = query_final_state[-1].h, b = response_final_state,
                transpose_b = True)
            self.losses = tf.losses.softmax_cross_entropy(
                onehot_labels = self.labels,
                logits = logits)
            self.mean_loss = tf.reduce_mean(self.losses, name="mean_loss")
            train_loss_summary = tf.summary.scalar('loss', self.mean_loss)
            self.training_summaries = tf.summary.merge(
                inputs = [train_loss_summary], name='train_monitor')

            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.mean_loss)

        elif self.mode=="valid" or self.mode=="eval":
            # Each sample has a fixed number of negative response
            # [batch_size x 20, rnn_dim]
            response_final_state = tf.matmul(response_final_state[-1].h, W)
            query_final_state = tf.reshape(
                tf.tile(query_final_state[-1].h, [1, self.args.ranksize]),
                [-1, self.args.hiddenSize])
            # [batch_size, batch_size x 20]
            logits = tf.reduce_sum(
                tf.multiply(
                    x = query_final_state,
                    y = response_final_state),
                axis = 1,
                keep_dims = True)
            logits = tf.reshape(logits, [-1, self.args.ranksize])
            # top_k percentage
            self.response_top_1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(
                    predictions = logits,
                    targets = self.targets,
                    k = 1,
                    name = 'prediction_in_top_1'),
                    dtype = tf.float32))
            self.response_top_3 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(
                    predictions = logits,
                    targets = self.targets,
                    k = 3,
                    name = 'prediction_in_top_3'),
                    dtype = tf.float32))
            self.response_top_5 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(
                    predictions = logits,
                    targets = self.targets,
                    k = 5,
                    name = 'prediction_in_top_5'),
                    dtype = tf.float32))


            top1_summary = tf.summary.scalar('valid_top1_of%s' % self.args.ranksize, self.response_top_1)
            top3_summary = tf.summary.scalar('valid_top3_of%s' % self.args.ranksize, self.response_top_3)
            top5_summary = tf.summary.scalar('valid_top5_of%s' % self.args.ranksize, self.response_top_5)
            self.evaluation_summaries = tf.summary.merge(
                inputs = [top1_summary, top3_summary, top5_summary],
                name='eval_monitor')

            self.outputs = (self.response_top_1,
                            self.response_top_3, self.response_top_5, logits)

        elif self.mode=="predict":
            # Restore variables from disk.
            if self.args.mode=="predict" and os.path.exists(self.modelDir):
                saver = tf.train.Saver()
                saver.restore(self.sess, self.bestModelPath)
                print("Model restored.")

                g1 = tf.Graph()
                cachesess = tf.Session(graph=g1)
                with cachesess.as_default():
                    with g1.as_default():
                        cachesaver = tf.train.import_meta_graph(os.path.join(self.cachePath, 'cache.ckpt.meta'))
                        ckpt_file = os.path.join(self.cachePath, 'cache.ckpt')
                        cachesaver.restore(cachesess, ckpt_file)
                        self.response_final_state = cachesess.run('cache_response_final_state:0')
                        self.response_seqs = cachesess.run('cache_response_seqs:0')
                        self.ranksize = self.response_final_state.shape[0]
                print("Cache restored...")


                query_final_state = tf.reshape(
                    tf.tile(query_final_state[-1].h, [1, self.ranksize]),
                    [-1, self.args.hiddenSize])

                logits = tf.reduce_sum(
                    tf.multiply(
                        x = query_final_state,
                        y = self.response_final_state),
                    axis = 1,
                    keep_dims = True)

                logits = tf.reshape(logits, [-1, self.ranksize])
                logits = tf.nn.softmax(logits)
                self.outputs = (logits)

        elif self.mode=="cache":
            response_final_state = tf.matmul(response_final_state[-1].h, W)
            self.outputs = (response_final_state)

        else:
            raise Exception("Unkown mode")



    def step(self, batch):
        """ Forward/training step operation.
        """
        # Feed the dictionary
        feedDict = {}
        ops = None

        if self.mode!="cache":
            feedDict[self.query_seqs] = batch.query_seqs
            feedDict[self.query_length] = batch.query_length

        if self.mode!="predict":
            feedDict[self.response_seqs] = batch.response_seqs
            feedDict[self.response_length] = batch.response_length

        if self.mode=="train":  # Training
            ops = (self.optOp, self.mean_loss, self.training_summaries)
            feedDict[self.labels] = np.eye(len(batch.query_seqs))
        elif self.mode=="valid" or self.mode=="eval": # Testing or Validating
            ops = (self.outputs, self.evaluation_summaries)
            feedDict[self.targets] = np.zeros((len(batch.query_seqs))).astype(int)
        elif self.mode=="cache":
            ops = self.outputs
        else: # Predicting
            ops = self.outputs
            feedDict[self.targets] = np.zeros((len(batch.query_seqs))).astype(int)
        # Return one pass operator
        return ops, feedDict


