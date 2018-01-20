# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

import os
import os.path
import jieba

"""
Opensource Chinese Movie Subtitle  Dialogue Corpus

"""

class XhjData:
    """
    """
    def __init__(self, dirName):
        """
        Args:
            dirName (string): data directory of xhj data
        """
        print('creating Xhj obj')

        if os.path.isfile(os.path.join(dirName, 'xhj.pkl')):
            print('loading from xhj.pkl')
            import pickle
            with open(os.path.join(dirName, 'xhj.pkl'),'rb') as f:
                self.conversations = pickle.load(f)
        else:
            self.conversations = []
            # fileName = os.path.join(dirName, 'xiaohuangji50w_nofenci.conv')
            fileName = os.path.join(dirName, 'test.conv')
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
            label= None
            for line in f:
                if lineID<100:
                    print(line)
                if lineID==0 or label=='E': # next dialogue
                    label = line[0]
                    content = line[2:].strip()
                    content = self.segment(content)
                    conversation = [{"text": [content.split('/')]}]
                else:
                    label = line[0]
                    if label!='E':
                        content = line[2:].strip()
                        content = self.segment(content)
                        conversation.append({"text":[content.split('/')]})
                    else:
                        self.conversations.append({"lines":conversation})
                lineID += 1
        return self.conversations


    def getConversations(self):
        return self.conversations

    def segment(self, content):
        seg_list = jieba.cut(content, cut_all=False)
        return "/".join(seg_list)