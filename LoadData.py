# coding: utf-8
# Author: Miracle Yoo
import codecs
import random
from midatapro3 import Pre_process_sentence
import jieba
from gensim.models import word2vec
from torch.utils.data import Dataset
import numpy as np


class PrepareData(object):
    def __init__(self,
                 char=False,
                 word2vec_model=None,
                 profession_dic_path="../reference_dic/_profession_dic.txt"):
        if word2vec_model is not None:
            self.model = word2vec.Word2Vec.load(word2vec_model)
        else:
            self.model = None 
        self.char = char
        self.profession_dic_path = profession_dic_path
        self.prep = Pre_process_sentence(profession_dic_path="../reference_dic/_profession_dic.txt")

    def preprocess_sentence(self, sentence):
        # 句子的预处理，返回结巴切词后的结果
        sentence = self.prep.dele_process(sentence)
        sentence = self.prep.standard_symbol_process(sentence)
        sentence = self.prep.delete_symbols_nums(sentence)
        seg = list(jieba.cut(sentence))
        if self.model is not None:
            for i in seg:
                try:
                    _ = self.model[i]
                except KeyError:
                    return None
        return seg

    def gen_vocab_dict(self, data_dict, test_dict):
        # 把训练集和测试集中的全部字或是词全放一起然后给一个编号，
        # 返回的vacab_dict的key是字/词，value是相应的index
        title      = data_dict.keys()
        title_test = test_dict.keys()
        question   = []
        for i in title:
            for j in data_dict[i]:
                question.extend(j)
            if self.char:
                question.extend(list(i))
            else:
                question.extend(self.preprocess_sentence(i))
        for i in title_test:
            for j in test_dict[i]:
                question.extend(j)
            if self.char:
                question.extend(list(i))
            else:
                question.extend(self.preprocess_sentence(i))
        question = list(set(question))
        vocab_dict = {}
        for i in list(range(len(question))):
            vocab_dict[question[i]] = i
        if " " not in question:
            vocab_dict[" "] = len(question)
        if self.char:
            print("==> Char dictionary build!")
        else:
            print("==> Vocab dictionary build!")
        return vocab_dict

    def gen_title(self, data_dict, test_dict):
        title = list(data_dict.keys())
        # title.extend(list(test_dict.keys()))
        title = [i for i in title if "模糊" not in i]
        title = list(set(title))
        return title

    def gen_data_dict(self, data_path=None, data_list=None):
        # 生成一个data_dict，key是主问题，value是相关用户问题的字/词数组
        if not data_path == None:
            with codecs.open(data_path,encoding='utf-8') as f:
                data_correct = [[i.split("||")[0], i.split("||")[1].strip('\n')] for i in f.readlines()]
        elif not data_list == None:
            data_correct = [[i.split("||")[0], i.split("||")[1].strip('\n')] for i in data_list]
        # 去掉用户问法模糊这一类
        data_correct = [i for i in data_correct if "模糊" not in i[1]]
        data_dict = {}
        # data_dict的结构是{主问题1:[query1,query2,...],主问题2:[query1,query2,...],...}
        for i in data_correct:
            if i[1] in data_dict.keys():
                data_dict[i[1]].append(i[0])
            else:
                data_dict[i[1]] = [i[0]]

        def preprocess(data_dict):
            # 把data_dict的query们转换为汉字的list或是jieba分词后的词语list
            for i in data_dict.keys():
                temp = []
                for j in range(len(data_dict[i])):
                    sentence = data_dict[i][j]
                    if self.char:
                        sentence = list(sentence)
                        temp.append(sentence)
                    else:
                        sentence = self.preprocess_sentence(sentence)
                        if sentence is not None:
                            if len(sentence) <= 20:
                                temp.append(sentence)
                data_dict[i] = temp
            return data_dict

        data_dict = preprocess(data_dict)
        return data_dict

    def load_triplet_data(self, data_dict):
        def gen_train_data(data_dict, title):
            positive = data_dict[title]
            train_data = []
            for j in data_dict.keys():
                if self.preprocess_sentence(j) is not None:
                    if j != title:
                        temp = self.preprocess_sentence(j)
                        processed_title = self.preprocess_sentence(title)
                        if processed_title is not None:
                            for i in positive:
                                train_data.append([i, processed_title, temp])
            return train_data

        trainData = []
        for i in data_dict.keys():
            trainData.extend(gen_train_data(data_dict, i))
        print("train data size:", len(trainData))
        random.shuffle(trainData)
        return trainData

    def load_cls_data(self, data_dict, title, train=True, augamentation=False):
        # title是所有主问题的一个数组，本函数作用是返回data_dict中所有问题的一个(问题,主问题index)对
        # 如果是训练时，把(主问题,主问题index)也加进去
        trainData = []
        for i in title:
            if i in data_dict.keys():
                for j in data_dict[i]:
                    trainData.append((j, title.index(i)))
                    if augamentation:
                        random.shuffle(j)
                        trainData.append((j, title.index(i)))
                if train:
                    if self.char:
                        trainData.append((list(i), title.index(i)))
                    else:
                        trainData.append((self.preprocess_sentence(i), title.index(i)))
        random.shuffle(trainData)
        return trainData
    
    def load_title_as_data(self, title):
        # 返回主问题的(主问题,主问题index)对
        trainData = []
        for i in title:
            trainData.append((i, title.index(i)))
        random.shuffle(trainData)
        return trainData


class BeibeiClassification(Dataset):
    # 输入数据是load_cls_data的输出，如果word2vec_model存在，返回的是(词向量,主问题index)，
    # 如果不存在，则返回的是()
    def __init__(self, train_data, word2vec_model=None, vocab_dict=None, stop_dic_path=None, char=False):
        self.train_data    = train_data
        self.vocab_dict    = vocab_dict
        self.char = char
        if self.char:
            self.seq_len   = 20
        else:
            self.seq_len   = 20
        if stop_dic_path is not None:
            self.stop_dict = [line.strip() for line in open(stop_dic_path, 'r').readlines()]
        else:
            self.stop_dict = None
        self.vocab = vocab_dict.keys()
        if word2vec_model is not None:
            self.model     = word2vec.Word2Vec.load('../models/beibei_gensim.model')
        else:
            self.model     = None

    def __getitem__(self, index):
        data, label = self.train_data[index]
        data      = [i for i in data if i in self.vocab]
        if self.stop_dict is not None:
            data  = [i for i in data if i not in self.stop_dict]
        if len(data) < self.seq_len:
            data += [" "] * (self.seq_len - len(data))
        else:
            data  = data[:self.seq_len]
        if self.model is not None:
            data  = self.model[data[:self.seq_len]]
        elif self.vocab_dict is not None:
            data  = np.array([self.vocab_dict[i] for i in data[:self.seq_len] if i in self.vocab_dict.keys()])
        else:
            print("errors")

        return data, label, self.train_data[index]

    def __len__(self):
        return len(self.train_data)


class BeibeiTriplet(Dataset):
    def __init__(self, train_data, train_num, word2vec_model=None, vocab_dict=None):
        self.train_num  = train_num
        self.vocab_dict = vocab_dict
        if word2vec_model is not None:
            self.model  = word2vec.Word2Vec.load('../models/beibei_gensim.model')
        else:
            self.model  = None
        self.train_data = train_data[:train_num]

    def __getitem__(self, index):
        data1, data2, data3 = self.train_data[index]
        if len(data1) < 20:
            data1 += [" "] * (20 - len(data1))
        if len(data2) < 20:
            data2 += [" "] * (20 - len(data2))
        if len(data3) < 20:
            data3 += [" "] * (20 - len(data3))
        if self.model is not None:
            data1 = self.model[data1[:20]]
            data2 = self.model[data2[:20]]
            data3 = self.model[data3[:20]]
        elif self.vocab_dict is not None:
            data1 = [self.vocab_dict[i] for i in data1[:20]]
            data2 = [self.vocab_dict[i] for i in data2[:20]]
            data3 = [self.vocab_dict[i] for i in data3[:20]]
        else:
            print("errors")

        return data1, data2, data3

    def __len__(self):
        return self.train_num


class BeibeiTripletRam(Dataset):
    def __init__(self, train_dict, train_num, word2vec_model=None, vocab_dict=None):
        self.train_dict = train_dict
        self.train_num  = train_num
        self.vocab_dict = vocab_dict
        if word2vec_model is not None:
            self.model  = word2vec.Word2Vec.load('../models/beibei_gensim.model')
        else:
            self.model  = None

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.train_num


class BalancedData(Dataset):
    def __init__(self, data_dict, title, opt, word2vec_model=None, vocab_dict=None, stop_dic_path=None):
        self.data_dict     = data_dict
        self.title         = title
        self.vocab_dict    = vocab_dict
        self.vocab         = list(vocab_dict.keys())
        self.char          = opt.USE_CHAR
        self.datakeys      = list(self.data_dict.keys())
        random.shuffle(self.datakeys)
        if self.char:
            self.seq_len   = 20
        else:
            self.seq_len   = 20
        if stop_dic_path is not None:
            self.stop_dict = [line.strip() for line in open(stop_dic_path, 'r').readlines()]
        else:
            self.stop_dict = None
        if word2vec_model is not None:
            self.model     = word2vec.Word2Vec.load('../models/beibei_gensim.model')
        else:
            self.model     = None

    def __getitem__(self, index):
        sent      = random.choice(self.data_dict[self.datakeys[index]])
        data      = [i for i in sent if i in self.vocab]
        label     = self.title.index(self.datakeys[index])
        if self.stop_dict is not None:
            data  = [i for i in data if i not in self.stop_dict]
        if len(data) < self.seq_len:
            data += [" "] * (self.seq_len - len(data))
        else:
            data  = data[:self.seq_len]
        if self.model is not None:
            data  = self.model[data[:self.seq_len]]
        elif self.vocab_dict is not None:
            data  = np.array([self.vocab_dict[i] for i in data[:self.seq_len] if i in self.vocab_dict.keys()])
        else:
            print("errors")

        return data, label, sent

    def __len__(self):
        return len(self.data_dict)