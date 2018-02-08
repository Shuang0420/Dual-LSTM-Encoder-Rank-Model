# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

"""
Main script. See README.md for more information

Use python 3
"""

from rankbot.rankbot import Rankbot
from rankbot.ranker import Ranker
from rankbot.ranktextdata import RankTextData
import tensorflow as tf
import numpy as np
import sys


# predict
pred_args_in = '--device gpu0 ' \
          '--modelTag test_xhj_2l_lr002_dr09_iniemb64_len50_vocab300_rank30 ' \
          '--skipLines ' \
          '--initEmbeddings ' \
          '--embeddingSource news_12g_baidubaike_20g_novel_90g_embedding_64.bin ' \
          '--embeddingSize 64 ' \
          '--vocabularySize 300 --maxLength 50 ' \
          '--filterVocab 3 ' \
          '--ranksize 30 ' \
          '--learningRate 0.002 --dropout 0.9 ' \
          '--saveEvery 500 ' \
          '--numEpochs 1000 ' \
          '--rootDir  /your/path/to/chatbot/Dual-LSTM-Encoder-Rank-Model ' \
          '--mode predict ' \
          '--datasetTag xiaohuangji --corpus xiaohuangji'.split()


# evaluation
eval_args_in = '--device gpu0 ' \
          '--modelTag test_xhj_2l_lr002_dr09_iniemb64_len50_vocab300_rank30 ' \
          '--skipLines ' \
          '--initEmbeddings ' \
          '--embeddingSource news_12g_baidubaike_20g_novel_90g_embedding_64.bin ' \
          '--embeddingSize 64 ' \
          '--vocabularySize 300 --maxLength 50 ' \
          '--filterVocab 3 ' \
          '--ranksize 30 ' \
          '--learningRate 0.002 --dropout 0.9 ' \
          '--saveEvery 500 ' \
          '--numEpochs 1000 ' \
          '--rootDir  /your/path/to/chatbot/Dual-LSTM-Encoder-Rank-Model ' \
          '--mode eval ' \
          '--datasetTag xiaohuangji --corpus xiaohuangji'.split()


if __name__ == "__main__":
    rankbot = Rankbot()
    if len(sys.argv) > 1 and sys.argv[1]=="eval":
        rankbot.main(pred_args_in)
        while True:
            print(rankbot.mainPredict(input("> "), mysess=rankbot.sess)[0])
    else:
        rankbot.main(eval_args_in)


