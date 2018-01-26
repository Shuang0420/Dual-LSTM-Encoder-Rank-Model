# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

import os
import os.path
import jieba

"""
Opensource FAQ Dialogue Corpus
Format: Q \t StandQ \t StandA

"""

class FaqaqData:
    """
    """
    def __init__(self, dirName):
        """
        Args:
            dirName (string): data directory of xhj data
        """
        print('creating FAQaq obj')

        if os.path.isfile(os.path.join(dirName, 'faqaq.pkl')):
            print('loading from faq.pkl')
            import pickle
            with open(os.path.join(dirName, 'faqaq.pkl'),'rb') as f:
                self.conversations = pickle.load(f)
        else:
            self.conversations = []
            fileName = os.path.join(dirName, 'faq_aq.conv')
            self.loadConversations(fileName)


    def loadConversations(self, fileName):
        """
        Args:
            fileName (str): file to load
        Return:
            list<dict<str>>: the extracted fields for each line
        """
        with open(fileName, 'r') as f:
            lineID = 0
            for line in f:
                if lineID<100:
                    print(line)
                parts = line.strip().split('\t', 1)
                if len(parts)==2:
                    content = self.segment(parts[0])
                    conversation = [{"text": [content.split('/')]}]
                    content = self.segment(parts[1])
                    conversation.append({"text":[content.split('/')]})
                    self.conversations.append({"lines":conversation})
                lineID += 1
        return self.conversations


    def getConversations(self):
        return self.conversations



    def segment(self, content):
        seg_list = jieba.cut(content, cut_all=False)
        return "/".join(seg_list)
