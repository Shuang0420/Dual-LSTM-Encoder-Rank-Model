# Dual-LSTM-Encoder-Rank-Model
Retrieval-Based Conversational Model
See **Retrieval-Based Models** part in blog [经典的端到端聊天模型](http://www.shuang0420.com/2017/10/05/经典的端到端聊天模型/) for details.

# Install
```$xslt
git clone https://github.com/Shuang0420/Dual-LSTM-Encoder-Rank-Model.git
cd Dual-LSTM-Encoder-Rank-Model
python setup.py build
python setup.py install
```

 # Usage
See examples at ``main.py`` for training and ``predict.py`` for inference.

**Arguments: **
```$xslt
usage: main.py [-h] [--mode MODE] [--restore] [--keepAll]
               [--modelTag MODELTAG] [--rootDir ROOTDIR] [--device DEVICE]
               [--seed SEED]
               [--corpus {cornell,opensubs,scotus,ubuntu,lightweight,xiaohuangji,wechat,chatterbot,baike}]
               [--datasetTag DATASETTAG] [--maxLength MAXLENGTH]
               [--filterVocab FILTERVOCAB] [--skipLines]
               [--vocabularySize VOCABULARYSIZE] [--ranksize RANKSIZE]
               [--train_frac TRAIN_FRAC] [--valid_frac VALID_FRAC]
               [--hiddenSize HIDDENSIZE] [--numLayers NUMLAYERS]
               [--initEmbeddings] [--embeddingSize EMBEDDINGSIZE]
               [--embeddingSource EMBEDDINGSOURCE] [--numEpochs NUMEPOCHS]
               [--saveEvery SAVEEVERY] [--batchSize BATCHSIZE]
               [--learningRate LEARNINGRATE] [--dropout DROPOUT]

optional arguments:
  -h, --help            show this help message and exit

Global options:
  --mode MODE           train or predict
  --restore             If True, restore previous model and continue training
  --keepAll             If this option is set, all saved model will be kept
                        (Warning: make sure you have enough free disk space or
                        increase saveEvery)
  --modelTag MODELTAG   tag to differentiate which model to store/load
  --rootDir ROOTDIR     folder where to look for the models and data
  --device DEVICE       'gpu' or 'cpu' (Warning: make sure you have enough
                        free RAM), allow to choose on which hardware run the
                        model
  --seed SEED           random seed for replication

Dataset options:
  --corpus {cornell,opensubs,scotus,ubuntu,lightweight,xiaohuangji,wechat,chatterbot,baike}
                        corpus on which extract the dataset..
  --datasetTag DATASETTAG
                        add a tag to the dataset (file where to load the
                        vocabulary and the precomputed samples, not the
                        original corpus). Useful to manage multiple versions.
                        Also used to define the file used for the lightweight
                        format.
  --maxLength MAXLENGTH
                        maximum length of the sentence (for input and output),
                        define number of maximum step of the RNN
  --filterVocab FILTERVOCAB
                        remove rarelly used words (by default words used only
                        once). 0 to keep all words.
  --skipLines           Generate training samples by only using even
                        conversation lines as questions (and odd lines as
                        answer, [2*i, 2*i+1]). Useful to train the network on
                        a particular person.
  --vocabularySize VOCABULARYSIZE
                        Limit the number of words in the vocabulary (0 for
                        unlimited)
  --ranksize RANKSIZE   Negative responses plus 1
  --train_frac TRAIN_FRAC
                        percentage of training samples
  --valid_frac VALID_FRAC
                        percentage of training samples

Network options:
  architecture related option

  --hiddenSize HIDDENSIZE
                        number of hidden units in each RNN cell
  --numLayers NUMLAYERS
                        number of rnn layers
  --initEmbeddings      if present, the program will initialize the embeddings
                        with pre-trained word2vec vectors
  --embeddingSize EMBEDDINGSIZE
                        embedding size of the word representation
  --embeddingSource EMBEDDINGSOURCE
                        embedding file to use for the word representation

Training options:
  --numEpochs NUMEPOCHS
                        maximum number of epochs to run
  --saveEvery SAVEEVERY
                        nb of mini-batch step before creating a model
                        checkpoint
  --batchSize BATCHSIZE
                        mini-batch size
  --learningRate LEARNINGRATE
                        Learning rate
  --dropout DROPOUT     Dropout rate (Dropout rate (keep probabilities)
```
