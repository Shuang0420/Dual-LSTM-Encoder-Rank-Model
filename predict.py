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

# args_in = '--device gpu0 ' \
#           '--modelTag test_xhj_2l_lr002_dr09_iniemb64_len50_vocab300_rank30 ' \
#           '--skipLines ' \
#           '--initEmbeddings ' \
#           '--embeddingSource news_12g_baidubaike_20g_novel_90g_embedding_64.bin ' \
#           '--embeddingSize 64 ' \
#           '--vocabularySize 300 --maxLength 50 ' \
#           '--filterVocab 3 ' \
#           '--ranksize 30 ' \
#           '--learningRate 0.002 --dropout 0.9 ' \
#           '--saveEvery 20 ' \
#           '--numEpochs 10 ' \
#           '--rootDir  /your/path/to/chatbot/Dual-LSTM-Encoder-Rank-Model ' \
#           '--mode predict ' \
#           '--datasetTag xiaohuangji --corpus xiaohuangji'.split()
args_in = '--device gpu0 ' \
          '--modelTag test_faq_2l_lr002_dr09_iniemb64_len50_vocab300_rank30 ' \
          '--skipLines ' \
          '--initEmbeddings ' \
          '--restore ' \
          '--embeddingSource news_12g_baidubaike_20g_novel_90g_embedding_64.bin ' \
          '--embeddingSize 64 ' \
          '--vocabularySize 0 --maxLength 50 ' \
          '--filterVocab 0 ' \
          '--ranksize 100 ' \
          '--learningRate 0.002 --dropout 0.9 ' \
          '--saveEvery 20 ' \
          '--numEpochs 100 ' \
          '--rootDir  /Users/xushuang/sf/chatbot/YanBot ' \
          '--mode predict ' \
          '--datasetTag faq --corpus faq'.split()

if __name__ == "__main__":
    rankbot = Rankbot()
    rankbot.main(args_in)
    while True:
        print(rankbot.mainPredict(input("> "), mysess=rankbot.sess))

