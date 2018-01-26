# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

from tqdm import tqdm  # Progress bar

import collections
import numpy as np
import os
import pickle
import random
import string
import jieba

from rankbot.corpus.cornelldata import CornellData
from rankbot.corpus.opensubsdata import OpensubsData
from rankbot.corpus.scotusdata import ScotusData
from rankbot.corpus.ubuntudata import UbuntuData
from rankbot.corpus.xhjdata import XhjData
from rankbot.corpus.wechatdata import WechatData
from rankbot.corpus.lightweightdata import LightweightData
from rankbot.corpus.chatterbot import ChatterbotData
from rankbot.corpus.baikedata import BaikeData
from rankbot.corpus.faqdata import FaqData
from rankbot.corpus.faqaqdata import FaqaqData


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        # nice summary of various sequence required by chatbot training.
        self.query_seqs = []
        self.response_seqs = []
        # self.xxlength：encoding阶段调用 dynamic_rnn 时的一个 input_argument
        self.query_length = []
        self.response_length = []


class TextData:
    """Dataset class
    """

    # OrderedDict because the first element is the default choice
    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData),
        ('scotus', ScotusData),
        ('ubuntu', UbuntuData),
        ('lightweight', LightweightData),
        ('xiaohuangji', XhjData),
        ('wechat', WechatData),
        ('chatterbot', ChatterbotData),
        ('baike', BaikeData),
        ('faq', FaqData),
        ('faqaq', FaqaqData)
    ])


    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        self.args = args

        # Path variables
        self._processPaths()

        # 1. Special tokens
        self.padToken = -1  # Padding
        self.goToken = -1   # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary
        # 2. Dictionary
        self.word2id = {} # {word：int}
        self.id2word = {} # {int：word}
        self.idCount = {} # Useful to filters the words，{word：count}
        # 3. Data lsit
        # train/valid/test: list<[input,target]>
        # question/input and response/target: list<int>
        self.predictingResponseSeqs = []
        self.predictingResponseLength = []
        self.predictingSamples = []
        self.trainingSamples = []
        self.validationSamples = []
        self.testingSamples = []


        # Load/create the conversations data
        self.loadCorpus()

        if len(self.predictingResponseSeqs) == 0:
            self.predictingSamples.extend(self.trainingSamples)
            self.predictingSamples.extend(self.validationSamples)
            self.predictingSamples.extend(self.testingSamples)

        # Plot some stats:
        self._printStats()

    def _processPaths(self):
        """Model/dataset related path

        Eg., Dataset related parameters
        （args.rootDir，args.corpus， args.datasetTag) = ('test', 'ubuntu', 'round3_5')
        Model related parameters
        （args.maxLength, args.filterVocab, args.vocabularySize）= （20， 1， 20K）
        self.corpusDir = 'test/data/ubuntu'
        self.fullSamplesPath = 'test/data/samples/dataset-ubuntu-round3_5.pkl'
        self.filteredSamplesPath = 'test/data/samples/dataset-ubuntu-round3_5-length20-filter1-vocaSize20000.pkl'
        """

        self.corpusDir = os.path.join(
            self.args.rootDir, 'data', self.args.corpus)

        targetPath = os.path.join(self.args.rootDir, 'data/samples',
                                  'dataset-{}'.format(self.args.corpus))
        if self.args.datasetTag:
            targetPath += '-' + self.args.datasetTag


        self.fullSamplesPath = targetPath + '.pkl'


        self.filteredSamplesPath = targetPath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )


    def _printStats(self):
        print('Loaded {}: {} words, {} training QA samples'.format(
            self.args.corpus, len(self.word2id), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)



    def loadCorpus(self):
        """Load/create the conversations data
            1. self.fullSamplePath
            2. self.filteredSamplesPath
        """
        print('filteredSamplesPath:%s' % self.filteredSamplesPath)
        # Try to construct the dataset from the preprocessed entry
        datasetExist = os.path.isfile(self.filteredSamplesPath)

        if not datasetExist:
            print('训练样本不存在。从原始样本数据集创建训练样本...')

            # 创建/读取原始对话样本数据集： self.trainingSamples
            print('fullSamplesPath:%s' % self.fullSamplesPath)
            datasetExist = os.path.isfile(self.fullSamplesPath)
            if not datasetExist:
                print('原始训练样本不存在。创建原始样本数据集...')
                # 1. Create corpus
                print('self.corpusDir: %s' % self.corpusDir)
                print('self.args.corpus: %s' % self.args.corpus)
                corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir)
                print(repr(corpusData))
                # 2. Preprocessing, extracting conversations
                self.createFullCorpus(corpusData.getConversations())
                # 3. Save full samples after simple preprocessing
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()

            # Postprocessing
            # 1. Filter uncommon (<=filterVocab) words and save common vocabSize words
            print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
                self.args.vocabularySize,
                self.args.filterVocab
            ))
            self.filterFromFull()

            # 2. Split data
            print('分割数据为 train, valid, test 数据集...')
            n_samples = len(self.trainingSamples)
            train_size = int(self.args.train_frac * n_samples)
            valid_size = int(self.args.valid_frac * n_samples)
            test_size = n_samples - train_size - valid_size

            print('n_samples=%d, train-size=%d, valid_size=%d, test_size=%d' % (
                n_samples, train_size, valid_size, test_size))
            self.shuffle()
            self.testingSamples = self.trainingSamples[-test_size:]
            self.validationSamples = self.trainingSamples[-valid_size-test_size : -test_size]
            self.trainingSamples = self.trainingSamples[:train_size]

            # 3. Save dataset
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath)
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0


    def saveDataset(self, filename):
        """Save samples to file。

        Including word dictionary and conversation samples

        Args:
            filename (str): pickle filename
        """
        with open(filename, 'wb') as handle:
            data = {
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainingSamples': self.trainingSamples
            }

            if len(self.validationSamples)>0:
                data['validationSamples'] = self.validationSamples
                data['testingSamples'] = self.testingSamples

            pickle.dump(data, handle, -1)  # Using the highest protocol available



    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']
            if 'validationSamples' in data:
                self.validationSamples = data['validationSamples']
                self.testingSamples = data['testingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']



    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """

        word = word.lower()

        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        else:
            # If not, we create a new entry
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId



    def extractConversation(self, conversation):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        if self.args.skipLines:
            # WARNING: The dataset won't be regenerated if the choice evolve
            # (have to use the datasetTag)
            step = 2
        else:
            step = 1

        # For each N-turn conversation, current line and next line is considered as a conversation
        # If skipLines，only 2*i and 2*i+1 is a 1-turn conversation sample, 2*i+1 won't be reused
        # Otherwise，the ith line is the response of (i-1)th conversation and the query of ith conversation

        # Iterate over all the lines of the conversation
        for i in tqdm_wrap(
                range(0, len(conversation['lines']) - 1, step),
                desc='Conversation',
                leave=False
        ):
            inputLine  = conversation['lines'][i]['text']
            targetLine = conversation['lines'][i+1]['text']

            inputWords  = []
            targetWords = []

            try:
                for i in range(len(inputLine)):
                    if len(inputLine[i])>0:
                        inputWords.append([])
                        for j in range(len(inputLine[i])):
                            inputWords[-1].append(self.getWordId(inputLine[i][j]))

                for i in range(len(targetLine)):
                    if len(targetLine[i])>0:
                        targetWords.append([])
                        for j in range(len(targetLine[i])):
                            targetWords[-1].append(self.getWordId(targetLine[i][j]))

            except:
                print('inputWords %s' % inputLine['text'])
                print('targetWords %s' % targetLine['text'])
            # Filter wrong samples (if one of the list is empty)
            if inputWords and targetWords:
                self.trainingSamples.append([inputWords, targetWords])



    def createFullCorpus(self, conversations):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence length or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding
        self.goToken = self.getWordId('<go>')    # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # UNK Word dropped from vocabulary

        # Preprocessing data
        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)


    def filterFromFull(self):
        """ Load the pre-processed full corpus and filter the vocabulary / sentences
        to match the given model options
        """

        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences. More for dialog processing.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer. Tricky:
                    if "question" is too long, keep the second half of question
                    if "answer" is too long, keep the first half of answer
            Return:
                list<int>: the list of the word ids of the sentence
            """

            merged = []

            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                # If the total length is not too big, we still can add one more sentence
                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else: # If the sentence is not used, neither are the words
                    try:
                        for w in sentence:
                            self.idCount[w] -= 1
                    except:
                        print('sentence: %s' % sentence)
            return merged

        newSamples = []

        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        for inputWords, targetWords in tqdm(
                self.trainingSamples, desc='Filter sentences:', leave=False):

            try:
                inputWords = mergeSentences(inputWords, fromEnd=True)
            except:
                print('inputWords: %s' % inputWords)
            targetWords = mergeSentences(targetWords, fromEnd=False)
            newSamples.append([inputWords, targetWords])




        # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?

        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        specialTokens = {
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        selectedWordIds = collections.Counter(self.idCount).most_common(
            self.args.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.args.filterVocab}
        selectedWordIds = selectedWordIds.union(specialTokens)


        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0
        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:
            if wordId in selectedWordIds:  # Update the word id
                word = self.id2word[wordId]
                # Update (word, wordId)
                # 1. Update wordId to newId
                newMapping[wordId] = newId
                # 2. newId <= wordId
                # 2-a. update word = id2word[wordId] ==>  id2word[newId]
                # 2-b. update word2id[word] = wordId ==> newId
                del self.id2word[wordId]
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else: # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]


        # Last step: replace old ids by new ones and filters empty sentences
        def replace_words(words):
            valid = False  # Filter empty sequences (all words are UNK)
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        del self.trainingSamples[:]

        for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            valid &= targetWords.count(self.unknownToken) == 0
            # Filter target with out-of-vocabulary target words ?

            if valid:
                self.trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple

        self.idCount.clear()



    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]

            batch.query_seqs.append(sample[0])
            batch.response_seqs.append(sample[1])
            batch.query_length.append(len(batch.query_seqs[-1]))
            batch.response_length.append(len(batch.response_seqs[-1]))

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.query_seqs[i]) <= self.args.maxLength
            assert len(batch.response_seqs[i]) <= self.args.maxLength

            # fill with padding to align batchSize samples into one 2D list
            batch.query_seqs[i] = batch.query_seqs[i] + [self.padToken] * (self.args.maxLength - len(batch.query_seqs[i]))
            batch.response_seqs[i]  = batch.response_seqs[i]  + [self.padToken] * (self.args.maxLength - len(batch.response_seqs[i]))

        return batch


    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)

        return batches



    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            sentence = [self.id2word[idx] for idx in sequence]

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        def detokenize(self, tokens):
            """Slightly cleaner version of joining with spaces.
            If there's single quote（for English，such as [Let 's], [some_name 's]），then add no space in front；
                else, add a space
            Args:
                tokens (list<string>): the sentence to print
            Return:
                str: the sentence
            """
            return ''.join([
                ' ' + t if not t.startswith('\'') and
                           t not in string.punctuation
                else t
                for t in tokens]).strip().capitalize()

        return self.detokenize(sentence)


    # 2. list<list<int>> ==> list<list<str>> ==> list<str>
    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)


    # 3.
    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Segment
        tokens = list(jieba.cut(sentence, cut_all=False))
        # tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch


    # 4.
    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually



    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


    #TODO: change to decoratoer
    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)


    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)


    def printBatch(self, batch):
        """Print a batch，for debug purpose
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.query_seqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.query_seqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.response_seqs, seqId=i)))


class RankTextData(TextData):

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Call base class initialisation
        super(RankTextData, self).__init__(args)
        del self.trainingSamples[:]
        self.valid_neg_idx = []
        self.test_neg_idx = []

    def _sampleNegative(self):
        """ For each sample in validation/testing dataset，generate [ranksize-1] negative responses

        For the purpose of computing Recall@k。

        """
        def npSampler(n):
            """Generate [n x 19] np.array
            """
            neg = np.zeros(shape = (n, self.args.ranksize-1))
            for i in range(self.args.ranksize-1):
                neg[:,i] = np.arange(n)
                np.random.shuffle(neg[:,i])
            findself = neg - np.arange(n).reshape([n, 1])
            findzero = np.where(findself==0)
            for (r, c) in zip(findzero[0], findzero[1]):
                x = np.random.randint(n)
                while x == r:
                    x = np.random.randint(n)
                neg[r, c] = x
            return neg.astype(int)

        n_valid = len(self.validationSamples)
        ##print('generating valid_neg_idx for %d' % n_valid)
        self.valid_neg_idx = npSampler(n_valid)
        n_test = len(self.testingSamples)
        ##print('generating test_neg_idx for %d' % n_test)
        self.test_neg_idx = npSampler(n_test)


    def _createEvalBatch(self, samples, dataset, neg_responses):
        """Create batch
        Args:
            samples (list<Obj>): list of samples, each sample is in form of[input, target]
            dataset: self.validationSamples or self.testingSamples
            neg_responses (2D array): neg_responses[i][j] the jth response for the ith sample

        Return:
            Batch: batch object
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        # using dynamic_rnn
        # time_major == False (default), shape = [batch_size, max_time, ...]

        for i in range(batchSize):

            sample = samples[i]

            batch.query_seqs.append(sample[0])
            batch.query_length.append(len(sample[0]))
            assert batch.query_length[-1] <= self.args.maxLength

            batch.response_seqs.append(sample[1])
            batch.response_length.append(len(sample[1]))
            assert batch.response_length[-1] <= self.args.maxLength

            for j in range(self.args.ranksize-1):
                sample = dataset[neg_responses[i][j]]
                batch.response_seqs.append(sample[1])
                batch.response_length.append(len(sample[1]))
                assert batch.response_length[-1] <= self.args.maxLength

            # pad sentence to simple length
            batch.query_seqs[i] = batch.query_seqs[i] + [self.padToken] * (
                self.args.maxLength  - len(batch.query_seqs[i]))

            for j in range(self.args.ranksize):
                batch.response_seqs[i*self.args.ranksize + j] = batch.response_seqs[i*self.args.ranksize + j] + [self.padToken] * (
                    self.args.maxLength - len(batch.response_seqs[i*self.args.ranksize + j]))

        return batch


    def getValidBatches(self):
        """Generate batch for model validation

       For each input sample，generate 1 true response and [ranksize] random wrong responses

        Return:
            batches: list<Batch>
        """

        # If valid_neg_idx is empty，caa _sampleNegative
        if self.valid_neg_idx == []:
            self._sampleNegative()

        batches = []
        n_valid = len(self.validationSamples)

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, n_valid, self.args.batchSize):
                yield (self.validationSamples[i:min(i + self.args.batchSize, n_valid)],
                       self.valid_neg_idx[i:min(i + self.args.batchSize, n_valid), :])

        for (samples, neg_responses) in genNextSamples():
            batch = self._createEvalBatch(samples, self.validationSamples, self.valid_neg_idx)
            batches.append(batch)

        return batches

    def getTestBatches(self):
        """Same as getValidBatches，just for testing data
        """

        if self.test_neg_idx == []:
            self._sampleNegative()


        batches = []
        n_test = len(self.testingSamples)

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, n_test, self.args.batchSize):
                yield (self.testingSamples[i:min(i + self.args.batchSize, n_test)],
                       self.test_neg_idx[i:min(i + self.args.batchSize, n_test), :])

        for (samples, neg_responses) in genNextSamples():
            batch = self._createEvalBatch(samples, self.testingSamples, self.test_neg_idx)
            batches.append(batch)
        return batches


    def getPredictBatchResponse(self):
        """Batch for prediction

        For each input input，use all responses in samples as response candidates

        Return:
            batch: Batch
        """
        batch = Batch()

        for sample in self.predictingSamples:
            sample = sample[1]
            assert len(sample) <= self.args.maxLength
            self.predictingResponseSeqs.append(sample)
            self.predictingResponseLength.append(len(sample))

            assert self.predictingResponseLength[-1] <= self.args.maxLength

            self.predictingResponseSeqs[-1] = self.predictingResponseSeqs[-1] + [self.padToken] * (
                self.args.maxLength - len(self.predictingResponseSeqs[-1]))

        batch.response_seqs = self.predictingResponseSeqs
        batch.response_length = self.predictingResponseLength

        return batch



    def getPredictBatchQuery(self, query):
        """Generate batch for prediction

        Args:
            query: query
        Return:
            batch: Batch
        """
        batch = Batch()

        if query == '':
            return None

        tokens = list(jieba.cut(query, cut_all=False))

        if len(tokens) > self.args.maxLength:
            return None

        wordIds = [self.getWordId(token, create=False) for token in tokens]

        query = wordIds


        # what to do if query is too long?
        if not query:
            return []

        batch.query_seqs.append(query)
        batch.query_length.append(len(query))
        assert batch.query_length[-1] <= self.args.maxLength
        # only one query
        batch.query_seqs[-1] = batch.query_seqs[-1] + [self.padToken] * (
            self.args.maxLength  - len(batch.query_seqs[-1]))

        return batch


    def getFinalEval(self):
        """Whole batch for final test including training, validation and testing samples
        Return:
            Batch: batch object
        """


        batches = []


        dataSamples = self.predictingSamples
        n_samples = len(dataSamples)


        # do the padding
        for i, sample in enumerate(dataSamples):
            batch = Batch()
            batch.query_seqs.append(sample[0])
            batch.query_length.append(len(sample[0]))
            assert batch.query_length[-1] <= self.args.maxLength

            batch.response_seqs.append(sample[1])
            batch.response_length.append(len(sample[1]))
            assert batch.response_length[-1] <= self.args.maxLength

            batch.query_seqs[-1] = batch.query_seqs[-1] + [self.padToken] * (
                self.args.maxLength  - len(batch.query_seqs[-1]))

            batch.response_seqs[-1] = batch.response_seqs[-1] + [self.padToken] * (
                self.args.maxLength - len(batch.response_seqs[-1]))

            batches.append(batch)

        return batches



    def seq2str(self, r, segment=False):
        if segment: return ' '.join([self.id2word[x] for x in r if x != 0])
        return ''.join([self.id2word[x] for x in r if x != 0])