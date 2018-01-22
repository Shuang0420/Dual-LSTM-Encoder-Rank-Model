# -*- coding: utf-8 -*-
# @Author      : Samuel Chan
# @FileName    : data_preprocess.py
# @E-mail      : samuelchan1205@gmail.com
# @Create Time : 2018/1/10 15:27

import os
import xlrd
import jieba


class preprocess:
    def __init__(self, fname):
        self.data_path = os.getcwd()#os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data"))
        self.QA_path = fname#os.path.join(self.data_path, "raw_data", "校招数据收集-智能回复机器人.xlsx")
        self.xlsx2text()

    def xlsx2text(self):
        '''
        读xlsx文件，根据sheet名字修改 workbook.sheet_by_name("sheet_name")
        根据小仙女要求写text文件
        :return: None
        '''
        workbook = xlrd.open_workbook(self.QA_path)
        standFAQ = workbook.sheet_by_name("标准问题，答案")
        fname = os.path.join(self.data_path, "standFAQ.txt")
        self.write_text(data=standFAQ, write_fname=fname)
        userQ = workbook.sheet_by_name("用户问题，标准问题")
        fname = os.path.join(self.data_path, "faq_full.txt")
        self.write_text(data=userQ, write_fname=fname)
        fname = os.path.join(self.data_path, "faq_seg.txt")
        self.write_text(data=userQ, write_fname=fname, cut=True)
        fname = os.path.join(self.data_path, "faq_classifier.txt")
        self.write_text(data=userQ, write_fname=fname, label="__label__faq", col_id=0, cut=True)

    def write_text(self, data, write_fname, label=None, col_id="all", sep="\t", cut=False):
        '''
        写text文件
        :param data: workbook data
        :param write_fname: 写文件路径
        :param label: str类型，默认None, 要添加的label
        :param col_id: int类型，默认"all", 抽取需要的列
        :param sep: str类型，默认"\t"，分隔符
        :param cut: True/False，默认False，是否分词
        :return: None
        '''
        f = open(write_fname, "w+", encoding="utf-8")
        for row in range(data.nrows):
            # skip header
            if row == 0:
                continue
            s = ""
            if col_id == "all":
                for col in range(data.ncols):
                    cell = data.cell(row, col)
                    val = cell.value.replace("\n", "")
                    if cut:
                        val = " ".join(list(jieba.cut(val, cut_all=False)))
                    s += val + sep
            else:
                val = data.cell(row, col_id).value.replace("\n", "")
                if cut:
                    val = " ".join(list(jieba.cut(val, cut_all=False)))
                s = label + sep + val
            s = s.rstrip(sep) + "\n"
            f.write(s)
        f.close()


import sys
fname = sys.argv[1]
p = preprocess(fname)
