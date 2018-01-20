# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

"""
Retrieval and ranking base chatbot. Referred to DeepQA repo.

Use python 3

"""

import argparse
import configparser
import datetime
import os
import tensorflow as tf
import time
import numpy as np

from tqdm import tqdm # Progress bar
from rankbot.ranktextdata import (TextData, RankTextData)
from rankbot.ranker import Ranker

import shutil


class Rankbot:
    """
    Retrieval-based chatbot
    """

    def __init__(self):

        self.args = None
        self.textData = None
        self.model = None

        self.modelDir = ''
        self.globStep = 0
        self.ckpt_model_saver = None
        self.best_model_saver = None
        self.saver = None
        self.best_model = []
        self.best_valid_loss = [float('inf'),
                                float('inf'),
                                float('inf')]

        self.sess = None

        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.BEST_MODEL_NAME_BASE = 'best_model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'


    @staticmethod
    def parseArgs(args):
        """
        Parse args
        Args:
            args (list<stir>): List of arguments.
        """

        parser = argparse.ArgumentParser()

        # Global options
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--mode', type=str, default="train",
                                help='train or predict')
        globalArgs.add_argument('--restore', action='store_true',
                            help='如果True, 在原有模型基础上训练')
        globalArgs.add_argument('--keepAll', action='store_true',
                                help='If this option is set, all saved model will be kept (Warning: make sure you have enough free disk space or increase saveEvery)')
        globalArgs.add_argument('--modelTag', type=str, default=None,
                                help='tag to differentiate which model to store/load')
        globalArgs.add_argument('--rootDir', type=str, default=None,
                                help='folder where to look for the models and data')
        globalArgs.add_argument('--device', type=str, default=None,
                                help='\'gpu\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')
        globalArgs.add_argument('--seed', type=int, default=None,
                                help='random seed for replication')

        # 数据相关的选项
        datasetArgs = parser.add_argument_group('Dataset options')
        datasetArgs.add_argument('--corpus', choices=TextData.corpusChoices(),
                                 default=TextData.corpusChoices()[0],
                                 help='corpus on which extract the dataset..')
        datasetArgs.add_argument('--datasetTag', type=str, default='',
                                 help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions. Also used to define the file used for the lightweight format.')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\')
        datasetArgs.add_argument('--maxLength', type=int, default=10,
                                 help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')
        datasetArgs.add_argument('--filterVocab', type=int, default=1,
                                 help='remove rarelly used words (by default words used only once). 0 to keep all words.')
        datasetArgs.add_argument('--skipLines', action='store_true',
                                 help='Generate training samples by only using even conversation lines as questions (and odd lines as answer, [2*i, 2*i+1]). Useful to train the network on a particular person.')
        datasetArgs.add_argument('--vocabularySize', type=int, default=20000,
                                 help='Limit the number of words in the vocabulary (0 for unlimited)')
        datasetArgs.add_argument('--ranksize', type=int, default=20,
                                 help='验证/测试集负样本大小+1')
        datasetArgs.add_argument('--train_frac', type=float, default = 0.8,
                                 help ='percentage of training samples')
        datasetArgs.add_argument('--valid_frac', type=float, default = 0.1,
                                 help ='percentage of training samples')

        # 模型结构选项
        nnArgs = parser.add_argument_group('Network options', 'architecture related option')
        nnArgs.add_argument('--hiddenSize', type=int, default=256,
                            help='number of hidden units in each RNN cell')
        nnArgs.add_argument('--numLayers', type=int, default=2,
                            help='number of rnn layers')
        nnArgs.add_argument('--initEmbeddings', action='store_true',
                            help='if present, the program will initialize the embeddings with pre-trained word2vec vectors')
        nnArgs.add_argument('--embeddingSize', type=int, default=256,
                            help='embedding size of the word representation')
        nnArgs.add_argument('--embeddingSource', type=str,
                            default="GoogleNews-vectors-negative300.bin",
                            help='embedding file to use for the word representation')

        # 模型训练设置
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--numEpochs', type=int, default=30,
                                  help='maximum number of epochs to run')
        trainingArgs.add_argument('--saveEvery', type=int, default=2000,
                                  help='nb of mini-batch step before creating a model checkpoint')
        trainingArgs.add_argument('--batchSize', type=int, default=256,
                                  help='mini-batch size')
        trainingArgs.add_argument('--learningRate', type=float, default=0.002,
                                  help='Learning rate')
        trainingArgs.add_argument('--dropout', type=float, default=0.9,
                                  help='Dropout rate (Dropout rate (keep probabilities)')

        return parser.parse_args(args)


    def main(self, args=None):
        """
        Launch the training and/or the predicting mode
        """
        print('Welcome to DeepRank!')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # General initialisation
        self.args = self.parseArgs(args)
        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()
        self.loadHyperParams()

        # Prepare data
        self.textData = TextData(self.args)
        self.evalData = RankTextData(self.args)


        if self.args.mode == "predict":
            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            )
            self.model_predictor = Ranker(self.args, self.textData, mode="predict", sess=self.sess)
            return


        # Prepare the model
        graph = tf.Graph()
        with tf.device(self.getDevice()):
            with graph.as_default():

                # allow_soft_placement = True: Allows backup device for non GPU-available operations (when forcing GPU)
                self.sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False)
                )

                with tf.name_scope('training'):
                    self.model_train = Ranker(self.args, self.textData, mode = "train")

                tf.get_variable_scope().reuse_variables()
                with tf.name_scope('validation'):
                    self.model_valid = Ranker(self.args, self.textData, mode = "valid")

                with tf.name_scope('evluation'):
                    self.model_test = Ranker(self.args,self.textData,  mode = "eval")
                    self.ckpt_model_saver = tf.train.Saver(name = 'checkpoint_model_saver')
                    self.best_model_saver = tf.train.Saver(name = 'best_model_saver')

                with tf.name_scope('cache'):
                    self.model_cache = Ranker(self.args, self.textData, mode = "cache")

                with tf.name_scope('prediction'):
                    self.model_predictor = Ranker(self.args, self.textData, mode = "predict", sess=self.sess)


                # Running session
                print('Initialize variables...')
                self.sess.run(tf.global_variables_initializer())

                # Saver/summaries
                graph_info = self.sess.graph
                self.train_writer = tf.summary.FileWriter(os.path.join(self.modelDir, 'train/'), graph_info)
                self.valid_writer = tf.summary.FileWriter(os.path.join(self.modelDir, 'valid/'), graph_info)
                self.saver = tf.train.Saver() # for restore

                # Restore previous model
                if self.args.mode == "train":
                    self.managePreviousModel(self.sess)

                # Initialize embeddings with pre-trained word2vec vectors
                if self.args.initEmbeddings:
                    self.loadEmbedding(self.sess)

                # Training
                if self.args.mode=="train":
                    self.mainTrain(self.sess)


    def managePreviousModel(self, sess):
        """ Restore or reset the model, depending of the parameters
        If the destination directory already contains some file, it will handle the conflict as following:
         * If --restore is not set, all present files will be removed (warning: no confirmation is asked) and the training
         restart from scratch (globStep & cie reinitialized)
         * Otherwise, it will depend of the directory content. If the directory contains:
           * No model files (only summary logs): works as a reset (restart from scratch)
           * The right model file (eventually some other): no problem, simply resume the training
        In any case, the directory will exist as it has been created by the summary writer
        Args:
            sess: The current running session
        """

        print('WARNING: ', end='')

        modelName = self._getModelName()

        if os.listdir(self.modelDir):
            if not self.args.restore:
                print('Reset: Destroying previous model at {}'.format(self.modelDir))
            # Analysing directory content
            elif os.path.exists(modelName): # Restore the model
                print('Restoring previous model from {}'.format(self.modelDir))
                self.saver.restore(sess, os.path.join(self.modelDir, 'best_model.ckpt'))
            else:
                print('No previous model found, but some files found at {}. Cleaning...'.format(self.modelDir))
                self.restore=False

            if not self.args.restore:
                shutil.rmtree(self.modelDir)
                # recreate
                graph_info = self.sess.graph
                self.train_writer = tf.summary.FileWriter(os.path.join(self.modelDir, 'train/'), graph_info)
                self.valid_writer = tf.summary.FileWriter(os.path.join(self.modelDir, 'valid/'), graph_info)


        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))


    def mainTrain(self, sess):
        """ Training loop
        Args:
            sess: The current tf session
        """

        print('Start training (press Ctrl+C to save and exit)...')

        try:
            batches_valid = self.evalData.getValidBatches()
            batches_test = self.evalData.getTestBatches()
            for e in range(self.args.numEpochs):
                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(
                    e+1, self.args.numEpochs, self.args.learningRate))

                batches = self.textData.getBatches()

                tic = datetime.datetime.now()
                for nextBatch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feedDict = self.model_train.step(nextBatch)
                    assert len(ops) == 3  # (training, loss)
                    _, loss, train_summaries = sess.run(ops, feedDict)
                    self.globStep += 1

                    # Output training status
                    if self.globStep % 100 == 0:
                        tqdm.write("----- Step %d -- CE Loss %.2f" % (self.globStep, loss))

                    # Checkpoint
                    if self.globStep % self.args.saveEvery == 0:
                        self.train_writer.add_summary(train_summaries, self.globStep)
                        self.train_writer.flush()

                        # validation pass
                        print('Evaluating on validation data ...')
                        self.valid_losses = [0,0,0]
                        for nextEvalBatch in tqdm(batches_valid, desc="Validation"):
                            ops, feedDict = self.model_valid.step(nextEvalBatch)
                            assert len(ops)==2
                            loss, eval_summaries = sess.run(ops, feedDict)
                            for i in range(3):
                                self.valid_losses[i] += loss[i]

                        self.valid_writer.add_summary(eval_summaries, self.globStep)
                        self.valid_writer.flush()

                        for i in range(3):
                            self.valid_losses[i] = self.valid_losses[i]/len(batches_valid)

                        print('validation, Recall_%s@(1,3,5) = %s' % (self.args.ranksize, self.valid_losses))
                        time.sleep(5)
                        if (len(self.best_model)==0) or (self.valid_losses[0] > self.best_valid_loss[0]):
                            print('best_model updated, with best accuracy :%s' % self.valid_losses)
                            self.best_valid_loss = self.valid_losses[:]
                            self._saveBestSession(sess)

                        self._saveCkptSession(sess)

                toc = datetime.datetime.now()
                print("Epoch %d finished in %s seconds" % (e, toc-tic))

            # Once training is over, try to run on testing data
            self.best_model_saver.restore(sess, self.best_model)
            self.test_losses = [0,0,0]
            for nextTestBatch in tqdm(batches_test, desc = "FinalTest"):
                ops, feedDict = self.model_test.step(nextTestBatch)
                assert len(ops)==2
                loss, _ = sess.run(ops, feedDict)
                for i in range(3):
                    self.test_losses[i] += loss[i]/len(batches_test)
            print('Final testing, Recall_%s@(1,3,5) = %s' % (self.args.ranksize, self.test_losses))


            self._saveCkptSession(sess)  # Ultimate saving before complete exit

            # Build cache
            batch = self.evalData.getPredictBatchResponse()
            self.best_model_saver.restore(sess, self.best_model)
            ops, feedDict = self.model_cache.step(batch)
            logits = sess.run(ops, feedDict)
            # Use last_state instead of output
            response_final_state = tf.Variable(logits, name='cache_response_final_state')
            response_seqs = tf.Variable(batch.response_seqs, name='cache_response_seqs')
            init = tf.global_variables_initializer()
            cachesaver = tf.train.Saver()
            cachePath = os.path.join(self.modelDir, 'cache')
            with tf.Session() as cachesess:
                cachesess.run(init)
                # Do some work with the model.
                # Save the variables to disk.
                save_path = cachesaver.save(cachesess, os.path.join(cachePath, 'cache.ckpt'))
                print("Cache saved in file: %s" % save_path)


            print("best model %s" % self.best_model)

        except (KeyboardInterrupt, SystemExit):

            print('Interruption detected, exiting the program...')



    def mainPredict(self, query, topK=1, mysess=None):
        batch = self.evalData.getPredictBatchQuery(query)

        ops, feedDict = self.model_predictor.step(batch)
        logits = mysess.run(ops, feedDict)
        idx = np.argmax(logits, axis=1)

        lres = list(np.argsort(-logits, axis=-1))[0]
        for i in lres[:topK]:
            res = self.model_predictor.response_seqs[int(i)]
        res = self.model_predictor.response_seqs[int(idx)]
        return self.evalData.seq2str(res)



    def _getModelName(self):
        """ Parse the argument to decide were to save/load the model
        This function is called at each checkpoint and the first time the model is load. If keepAll option is set, the
        globStep value will be included in the name.
        Return:
            str: The path and name were the model need to be saved
        """
        modelName = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        if self.args.keepAll:  # We do not erase the previously saved model by including the current step on the name
            modelName += '-' + str(self.globStep)
        return modelName + self.MODEL_EXT


    def _saveCkptSession(self, sess):
        """ Save session

        Args:
            sess: The current tf session
        """
        tqdm.write('Saving checkpoint (don\'t stop the run)...')
        tqdm.write('Validation, Recall_%s@(1,3,5) = %s' % (self.args.ranksize, repr(self.valid_losses)))
        self.saveHyperParams()

        # checkpoint name
        model_name = self._getModelName()
        print("save path: %s" % (model_name))
        self.ckpt_model_saver.save(sess, model_name)
        tqdm.write('Checkpoint saved.')


    def _saveBestSession(self, sess):
        """ Save best session

        Args:
            sess: The current tf session
        """
        tqdm.write('Save new BestModel (don\'t stop the run)...')
        self.saveHyperParams()

        # bestmodel checkpoint name
        model_name = os.path.join(self.modelDir, self.BEST_MODEL_NAME_BASE)
        model_name = model_name + self.MODEL_EXT

        self.best_model = self.best_model_saver.save(sess, model_name)
        tqdm.write('Best Model saved.')


    def loadHyperParams(self):
        """ Load the some values associated with the current model
        """
        # Compute the current model path
        self.modelDir = os.path.join(self.args.rootDir, self.MODEL_DIR_BASE)
        if self.args.modelTag:
            print("modelTag=%s" % self.args.modelTag)
            self.modelDir += '-' + self.args.modelTag
        print("modelDir=%s" % self.modelDir)

        # If there is a previous model, restore some parameters
        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if os.path.exists(configName):
            config = configparser.ConfigParser()
            config.read(configName)

            # Restoring the the parameters
            self.globStep = config['General'].getint('globStep')
            self.args.corpus = config['General'].get('corpus')

            self.args.datasetTag = config['Dataset'].get('datasetTag')
            self.args.maxLength = config['Dataset'].getint('maxLength')
            self.args.filterVocab = config['Dataset'].getint('filterVocab')
            self.args.skipLines = config['Dataset'].getboolean('skipLines')
            self.args.vocabularySize = config['Dataset'].getint('vocabularySize')

            self.args.hiddenSize = config['Network'].getint('hiddenSize')
            self.args.numLayers = config['Network'].getint('numLayers')
            self.args.initEmbeddings = config['Network'].getboolean('initEmbeddings')
            self.args.embeddingSize = config['Network'].getint('embeddingSize')
            self.args.embeddingSource = config['Network'].get('embeddingSource')


    def saveHyperParams(self):
        """ Save the params of the model
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['globStep']  = str(self.globStep)
        config['General']['corpus'] = str(self.args.corpus)

        config['Dataset'] = {}
        config['Dataset']['datasetTag'] = str(self.args.datasetTag)
        config['Dataset']['maxLength'] = str(self.args.maxLength)
        config['Dataset']['filterVocab'] = str(self.args.filterVocab)
        config['Dataset']['skipLines'] = str(self.args.skipLines)
        config['Dataset']['vocabularySize'] = str(self.args.vocabularySize)

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args.hiddenSize)
        config['Network']['numLayers'] = str(self.args.numLayers)
        config['Network']['initEmbeddings'] = str(self.args.initEmbeddings)
        config['Network']['embeddingSize'] = str(self.args.embeddingSize)
        config['Network']['embeddingSource'] = str(self.args.embeddingSource)

        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args.learningRate)
        config['Training (won\'t be restored)']['batchSize'] = str(self.args.batchSize)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)

        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)


    def getDevice(self):
        """ Parse the argument to decide on which device run the model
        Return:
            str: The name of the device on which run the program
        """
        if self.args.device == 'cpu':
            return '/cpu:0'
        elif self.args.device == 'gpu0':
            return '/gpu:0'
        elif self.args.device == 'gpu1':
            return '/gpu:1'
        elif self.args.device is None:
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None


    def loadEmbedding(self, sess):
        """ Initialize embeddings with pre-trained word2vec vectors
        Will modify the embedding weights of the current loaded model
        Uses the GoogleNews pre-trained values (path hardcoded)
        """
        with tf.name_scope('embedding_layer'):
            embedding = tf.get_variable('embedding',
                                        [len(self.textData.word2id), self.args.embeddingSize])


        # # Disable training for embeddings
        # variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        # variables.remove(em_in)
        # variables.remove(em_out)

        # If restoring a model, we can leave here
        if self.globStep != 0:
            return

        # New model, we load the pre-trained word2vec data and initialize embeddings
        embeddings_path = os.path.join(self.args.rootDir, 'data', 'embeddings', self.args.embeddingSource)
        embeddings_format = os.path.splitext(embeddings_path)[1][1:]
        print("Loading pre-trained word embeddings from %s " % embeddings_path)
        with open(embeddings_path, "rb") as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vector_size
            initW = np.random.uniform(-0.25,0.25,(len(self.textData.word2id), vector_size))
            for line in tqdm(range(vocab_size)):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode('utf-8')
                        break
                    if ch != b'\n':
                        word.append(ch)
                if word in self.textData.word2id:
                    if embeddings_format == 'bin':
                        vector = np.fromstring(f.read(binary_len), dtype='float32')
                    elif embeddings_format == 'vec':
                        vector = np.fromstring(f.readline(), sep=' ', dtype='float32')
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)
                    initW[self.textData.word2id[word]] = vector
                else:
                    if embeddings_format == 'bin':
                        f.read(binary_len)
                    elif embeddings_format == 'vec':
                        f.readline()
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)

        # PCA Decomposition to reduce word2vec dimensionality
        if self.args.embeddingSize < vector_size:
            U, s, Vt = np.linalg.svd(initW, full_matrices=False)
            S = np.zeros((vector_size, vector_size), dtype=complex)
            S[:vector_size, :vector_size] = np.diag(s)
            initW = np.dot(U[:, :self.args.embeddingSize], S[:self.args.embeddingSize, :self.args.embeddingSize])

        # Initialize input and output embeddings
        sess.run(embedding.assign(initW))