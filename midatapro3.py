# coding: utf-8
# Author: Miracle Yoo


"""
This module aims to solve almost all the fundamental but necessary while pretty troublesome problems.
The module is sepatated into several parts, each of them is packed into a class.
Process_knowledge is a class which contains functions who can do almost all the process for the "knowledge.xlsx" file.
Pre_process_sentence is a class which contains function who can do the necessary preprocessing, for instance, tradition->simplified Chinese.
Build_IDF_corpus can build a customized IDF corpus according to the Beibei corpus.
Accuracy_prediction class is desined to meet the needs that we need to compute the simiarity of two sentences, based on their cosine distance.
Test_process now can only handle the affairs of the test excel directly received from the company, we turn it into a more elegant form.
Build_word2vec_model class mainly is mainly made up of functions which is frequently used in the word embedding step.

# Author: Miracle Yoo & Haizi, or Zhang Zhongyang & Sun Haohai.

"""

import pandas as pd
import numpy as np
from scipy import spatial
import pickle
import jieba
import jieba.analyse
import time
# import opencc
import re
import math
import os
from gensim.models import word2vec,KeyedVectors

class Process_knowledge(object):
    """
        本模块的作用是对贝贝网提供的knowledge.xlsx进行各种必要的处理。
    """
    def __init__(self, in_filename = "../source/knowledge.xlsx"):
        self.df = pd.read_excel(in_filename)

    def build_re_model(self, startcol = 7, out_filename=u'../source/knowledge_for_matching.xlsx'):
        # 本函数的作用是把knowledge中的其他问题处理成可供匹配使用的re的model。
        dfc1 = self.df.copy()
        for i in range(dfc1.shape[0]):
            droped_sentence = dfc1.iloc[i].dropna()
            for j in range(startcol, len(droped_sentence)):
                cell = droped_sentence[j]
                if cell != np.nan and pd.isnull(cell) == False:
                    if cell.startswith('|'):
                        cell = cell[1:]
                    if cell.endswith('|'):
                        cell = cell[:-1]
                    cell = '(' + cell.strip() + ')'
                    new_cell = ''
                    for s in cell:
                        if s == '*':
                            new_cell += ').*('
                        elif s == '?' or s == u'？' or s == u'！':
                            pass
                        else:
                            new_cell += s
                    dfc1.iloc[i, j] = new_cell
        dfc1.to_excel(out_filename, index=False, encoding='utf-8')

    def print_every_ques_in_a_sep_sentence(self, out_filename=u'../source/pure_ques_sep_enter.txt'):
        # 本函数的作用是把knowledge中所有的主问题和问题分别提出来，每个问题（无论主副）打成一行
        df_pure_question = self.df.copy()
        df_pure_question.pop(u'答案')
        df_pure_question.pop(u'分类')
        with open(out_filename, 'w+') as output:
            for i in range(df_pure_question.shape[0]):
                for j in range(df_pure_question.shape[1]):
                    if j not in (1, 2, 3, 4):
                        content = df_pure_question.iloc[i, j]
                        if pd.isnull(content) == False and content != '':
                            if type(content).__name__ != "unicode":
                                content = str(content)
                            content = re.sub('[\r\n\t\|\*]', '', content)
                            content = content.encode('utf-8')
                            output.write(content + '\n')

    def print_all_same_main_ques_to_one_sentence(self,
                                                 out_filename=u'../source/pure_ques_in_a_sentence_sep_escape.txt'):
        # 本函数的作用是把knowledge中所有的主问题和其对应的分问题提出来，每个对应打成一行，用于生成自定义IDF_corpus
        df_pure_question = self.df.copy()
        df_pure_question.pop(u'答案')
        df_pure_question.pop(u'分类')
        with open(out_filename, 'w+') as output:
            for i in range(df_pure_question.shape[0]):
                for j in range(df_pure_question.shape[1]):
                    if j not in (1, 2, 3, 4):
                        content = df_pure_question.iloc[i, j]
                        if pd.isnull(content) == False and content != '':
                            if type(content).__name__ != "unicode":
                                content = str(content)
                            content = re.sub('[\r\n\t\|\*]', '', content)
                            content = content.encode('utf-8')
                            output.write(content + ' ')
                output.write('\n')

    def add_str(self, counter1, full_sent, nrow, all_parts, sentences_useful):
        part_sent = full_sent
        for counter2 in range(len(all_parts[counter1])):
            part_sent += all_parts[counter1][counter2]
            if counter1 < len(all_parts) - 1:
                self.add_str((counter1 + 1), part_sent, nrow, all_parts, sentences_useful)
                part_sent = full_sent
            else:
                sentences_useful[nrow].append(part_sent)
                part_sent = full_sent

    def build_sentences_useful(self, to_excel=True, out_filename='../source/pure_sentences_useful'):
        """
        本模块的作用是创建一个根据*和|拆分出的一个个句子
        to_excel:True:output a file in format xlsx.  False:output a txt file. Defalt is True.
        in_name :The name of the input file,defalt is '../source/knowledge.xlsx'.
        out_name:The name of the output file,defalt is '../source/pure_sentences_useful'.
        """
        pure_ques = self.df.iloc[:, 7:]
        sentences_useful = [[] for counter0 in range(pure_ques.shape[0])]
        for nrow in range(pure_ques.shape[0]):
            unnan_part = pure_ques.iloc[nrow].dropna()
            for ncol in range(len(unnan_part)):
                cell = pure_ques.iloc[nrow, ncol]
                star_parts = cell.split('*')
                all_parts = []
                for part in star_parts:
                    all_parts.append(part.split('|'))
                full_sent = ''
                self.add_str(0, full_sent, nrow, all_parts, sentences_useful)
        dataframesent = pd.DataFrame(sentences_useful)
        dataframesent.fillna(value=np.nan, inplace=True)
        if to_excel == False:
            txt_filename = out_filename + '.txt'
            dataframesent.to_csv(txt_filename, sep=' ', encoding='utf-8', index=False, header=False)
        else:
            excel_filename = out_filename + '.xlsx'
            dataframesent.to_excel(excel_filename)


class Pre_process_sentence(object):
    """
    本类的作用是对一个给定的句子进行各种预处理。
    extract: 是否选择提取tf-idf前列区分度高的词语，默认为True
    profession_dic_path: 专业词典路径，用作用户自定义词典
    stop_dic_path: 停词路径
    IDF_corpus_path: IDF_corpus路径，用作用户自定义IDF
    """

    def __init__(self, profession_dic_path='../reference_dic/_profession_dic.txt',
                 stop_dic_path='../reference_dic/_Filter_dic.txt',
                 IDF_corpus_path='../reference_dic/IDF_corpus.txt', extract=True):
        self.stop_dic_path = stop_dic_path
        self.profession_dic_path = profession_dic_path
        self.IDF_corpus_path = IDF_corpus_path
        self.extract = extract

    def dele_process(self, sentence):
        """
        #本模块的作用是对特殊汉字“了”“的”进行特殊处理
        """
        if (r"的" in sentence) and (not (r"的确" in sentence)) and (not (r"的士" in sentence)) \
                and (not (r"的哥" in sentence)) and (not (r"的的确确" in sentence)):
            sentence = sentence.replace(r'的', '')
        if (r"了" in sentence) and (not (r"了解" in sentence)) and (not (r"了结" in sentence)) \
                and (not (r"了无" in sentence)) and (not (r"了却" in sentence)) and (not (r"了不起" in sentence)):
            sentence = sentence.replace(r'了', '')
        return sentence

    def standard_symbol_process(self, sentence):
        """
        # 符号处理,将有特殊标记的外文直接剔除。我们最后再将「」『』这些符号替换成引号，顺便删除空括号，这个函数不参与总调用
        """
        p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
        p2 = re.compile(r'[（\(][\s]*[）\)]')
        p3 = re.compile(r'[「『]')
        p4 = re.compile(r'[」』]')
        sentence = p1.sub(r'\2', sentence)
        sentence = p2.sub(r'', sentence)
        sentence = p3.sub(r'“', sentence)
        sentence = p4.sub(r'”', sentence)
        return sentence

    def delete_symbols_nums(self, sentence):
        return re.sub(r'[\r\n\t\d?\!\,\.\:\"\'\|\\：！，、；\-_ 。“”‘’？~"]', '', sentence).strip()

    def tradition_2_simplified(self, sentence):
        # 繁简转换
        return sentence#opencc.convert(sentence, config='t2s.json')

    def seg_sentence(self, sentence, use_stop=True):
        # 对句子进行分词
        jieba.load_userdict(self.profession_dic_path)
        sentence_seged = jieba.cut(sentence.replace("\t", " ").replace("\n", " ").strip())
        if use_stop == True:
            stopwords = [line.strip() for line in open(self.stop_dic_path, 'r').readlines()]  # 这里加载停用词的路径
            outstr = ''
            outlist = []
            for word in sentence_seged:
                if word not in stopwords:
                    if word != '\t':
                        outstr += word
                        outstr += " "
                        outlist.append(word)
        else:
            outstr = ' '.join(sentence_seged)
            outlist = sentence_seged
        return outstr, outlist

    def get_the_setence_key_word(self, query):
        # 问句关键词提取
        jieba.analyse.set_stop_words(self.stop_dic_path)
        jieba.analyse.set_idf_path(self.IDF_corpus_path)
        tags = jieba.analyse.extract_tags(query, topK=5)
        return tags

    def do_all_the_process_and_get_the_key_word(self, sentence):
        # 执行本类中前面所有函数，返回一个句子的关键词数组
        sentence = self.delete_symbols_nums(sentence)
        sentence = self.dele_process(sentence)
        sentence = self.tradition_2_simplified(sentence)
        sentence_parsed_str, sentence_parsed_list = self.seg_sentence(sentence)
        key_words = self.get_the_setence_key_word(sentence_parsed_str)
        if self.extract:
            return key_words
        else:
            return sentence_parsed_list

    def quick_process(self, sentence):
        # 快速预处理，不进行抽取
        sentence = self.delete_symbols_nums(sentence)
        sentence = self.dele_process(sentence)
        sentence = self.tradition_2_simplified(sentence)
        sentence_parsed_str, sentence_parsed_list = self.seg_sentence(sentence)
        return sentence_parsed_list


class Build_IDF_corpus(object):
    """
    # 用于生成自定义的IDF_corpus
    # 结合Process_knowledge类中的print_all_same_main_ques_to_one_sentence使用
    """
    def __init__(self, raw_filename, IDF_out_filename = u'../reference_dic/IDF_corpus.txt'):
        super(Build_IDF_corpus, self).__init__()
        self.raw_filename = raw_filename
        self.IDF_out_filename = IDF_out_filename

    def start_to_build_IDF_corpus(self):
        # 根据raw_filename中提到的语料库，建立自定义的IDF corpus，以供jieba分词时候使用
        all_dict = {}
        total = 0
        with open(self.raw_filename ,'r') as sentences, open(self.IDF_out_filename,'w+') as IDFs:
            for sentence in sentences:
                temp_dict = {}
                total += 1
                cut_sentence = jieba.cut(sentence, cut_all=False)
                condition = lambda t: t != " "
                cut_sentence = list(filter(condition, cut_sentence))
                for word in cut_sentence:
                    temp_dict[word] = 1
                for key in temp_dict:
                    num = all_dict.get(key, 0)
                    all_dict[key] = num + 1
            for key in all_dict:
                if all_dict[key]!=total:
                    w = key
                    p = '%.10f' % (math.log10(total/(all_dict[key] + 1)))
                    IDFs.write("%s %s%c"%(w,p,'\n'))


class Accuracy_prediction(object):
    """
    本模块的作用是各种测试相关函数。
    """
    def __init__(self,model_path='../models/beibei_gensim.model', num_features = 100):
        # 初始化
        super(Accuracy_prediction, self).__init__()
        self.model_dir = os.path.split(model_path)[0]
        self.model_path = model_path
        self.model = word2vec.Word2Vec.load(model_path)
        self.num_features = num_features

    def avg_feature_vector(self, sentence):
        # 输入是一个问句，输出是这个问句对应的特征数组(维数等于num_features)
        model=self.model
        mp = Pre_process_sentence()
        words = mp.quick_process(sentence) #mp.do_all_the_process_and_get_the_key_word(sentence)
        feature_vec = np.zeros((self.num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in model.wv.vocab:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def get_sim_btw_sentences(self,sent1,sent2):
        # 直接输入两个问句，输出是它们的相似度
        s1_afv = self.avg_feature_vector(sent1)
        s2_afv = self.avg_feature_vector(sent2)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        return sim

    def get_the_best_pair_by_look_up_table(self, sentence, data_path = u'../source/avg_feature_vector_of_knowledge.dat'):
        # 通过查表的方式，实现迅速确定最相似的主问题的目标
        know_feature_vec = pickle.load(open(data_path, "rb"), encoding='latin1')
        sent_afv = self.avg_feature_vector(sentence)
        ques_num = len(know_feature_vec)
        max_sim = 0; max_sim_counter = 0
        for i in range(ques_num):
            sim = 1 - spatial.distance.cosine(sent_afv, know_feature_vec[i])
            if sim > max_sim:
                max_sim = sim
                max_sim_counter = i
        return max_sim_counter

    def build_avg_feature_vector_of_knowledge(self, output_path = '../source/avg_feature_vector_of_knowledge.dat', input_path = '../source/main_ques.txt'):
        #构建模型的主问题向量并保存
        vecs = []
        with open(input_path,'r') as sentences:
            for sentence in sentences:
                vecs.append(self.avg_feature_vector(sentence))
        pickle.dump(vecs, open(output_path, "wb"), True)

    def translate_counter_to_main_ques(self, counter):
        # 输入是主问题的行数，输出是相应的主问题，和get_the_best_pair_by_look_up_table配合使用
        with open("../source/main_ques.txt") as mf:
            return mf.readlines()[counter]

    def load_dic_short_query(self,  dic_short_query_path='../source/short_query_and_its_main_ques.dat'):
        # 载入频繁出现的客户问题和其对应主问题的字典
        return pickle.load(open(dic_short_query_path, "rb"))

    def build_short_query_dic(self,dic_short_query_path='../source/short_query_and_its_main_ques.dat', test_path='../test/processed_test_txt/all_test.txt',):
        # 本函数的作用是提取出用户问题中频繁出现（>3次）的问题和其对应的主答案，结果导出到dic_short_query_path中。
        short_query_list=[]
        with open(test_path,'r') as tp:
            for line in tp:
                if line.strip() != '':
                    (main_ques, query)=line.rstrip('\n').strip().split('||')
                    if len(query) < 5:
                        short_query_list.append(query+'||'+main_ques)
        short_query_set = set(short_query_list)
        dic_short_query={}
        for item in short_query_set:
            if short_query_list.count(item) > 3:
                dic_short_query[item.split('||')[0]]=item.split('||')[1]
        pickle.dump(dic_short_query, open(dic_short_query_path, "wb"), True)

    def integration_testing(self, dic_short_query_path='../source/short_query_and_its_main_ques.dat', test_path='../test/processed_test_txt/all_test.txt', main_vec_path = '../source/avg_feature_vector_of_knowledge.dat'):
        # 集成测试函数。此函数提供的功能是直接的对全体测试集的测试。当前使用方法为word2vec和高频问题匹配。
        query_num=0
        true_pred_num = 0
        dic_short_query = self.load_dic_short_query(dic_short_query_path)
        time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        with open(test_path,'r') as tp, open('../results/'+time_now+'-result-false.txt','w+') as frt, open('../results/'+time_now+'-result-true.txt', 'w+') as trt:
            for line in tp:
                query_num += 1
                if line.strip() != '':
                    (main_ques, query)=line.rstrip('\n').strip().split('||')
                    max_sim = 0
                    got = ''
                    if query in dic_short_query.keys():
                        max_sim=1
                        got = dic_short_query[query]
                    else:
                        got = self.translate_counter_to_main_ques(self.get_the_best_pair_by_look_up_table(query, main_vec_path)).strip()
                    if got == main_ques:
                        true_pred_num += 1
                        trt.write("Now the accuracy is {},the query is {}\n\
                        the main question is {} ,and you got {}".format(true_pred_num / query_num, query, main_ques,got))
                        trt.write("the accuracy is {}".format(true_pred_num / query_num))
                        print("True!!!")
                        print("Now the accuracy is {},the query is {}\n\
                        the main question is {} ,and you got {}".format(true_pred_num / query_num, query, main_ques,got))
                        print("the accuracy is {}".format(true_pred_num / query_num))
                    else:
                        print("False...",frt)
                        frt.write("Now the accuracy is {},the query is {}\n\
                        the main question is {} ,and you got {}".format(true_pred_num / query_num, query, main_ques,got))
                        frt.write("the accuracy is {}".format(true_pred_num / query_num))
                        print("Now the accuracy is {},the query is {}\n\
                        the main question is {} ,and you got {}".format(true_pred_num / query_num, query, main_ques,got))
                        print("the accuracy is {}".format(true_pred_num / query_num))



class Test_process(object):
    """
    本类的作用是对贝贝网提供的日常log转化为测试集之前的必要处理
    """
    def __init__(self, in_filename):
        super(Test_process, self).__init__()
        self.in_filename = in_filename
        self.suffix = os.path.splitext(os.path.split(self.in_filename)[1])[1]
        if self.suffix == '.xlsx':
            self.in_file = pd.read_excel(in_filename)
        elif self.suffix == '.txt' or in_filename.split('.')[-1] == '.csv':
            self.in_file = pd.read_csv(in_filename)
        self.out_filename = os.path.split(self.in_filename)[0]+'/processed_test_txt/'+os.path.splitext(\
            os.path.split(self.in_filename)[1])[0]+'_processed.txt'

    def get_the_unnan_part(self):
        # 由于给出的测试里面有很多的没有主问题对应的问题，所以这里就只提出来有着主问题对应的问题并转为“问题||主问题”的形式以便使用
        standard_ques_list = self.in_file[u'标准问题'].dropna()
        query_list = self.in_file[u'问题']
        test_file=[]
        for key_num in standard_ques_list.keys():
            real_answer = standard_ques_list[key_num]
            query = query_list[key_num]
            processed_test = re.sub(r'[\r\n\t\d?\!\,\.\:\"\'\|\\：！，、； 。“”‘’？~"]', '', query).strip()
            if processed_test != '':
                str_realans_query=real_answer+'||'+processed_test
                test_file.append(str_realans_query)
        test_file_str = '\n'.join(test_file)
        with open(self.out_filename,'w+') as out_file:
            out_file.write(test_file_str)
            print("Process is finished! File is saved in the processed_test_txt directory.")

class Build_word2vec_model(object):
    """
    本模块主要功能是提供word2vec相关的操作。
    """
    def __init__(self):
        super(Build_word2vec_model, self).__init__()

    def load_model(self, model_path='../models/beibei_gensim.model'):
        # 加载以.model格式储存的模型
        model = word2vec.Word2Vec.load(model_path)
        self.model = model
        return self.model

    def load_bin_model(self, model_path='../models/beibei_gensim.model.bin'):
        # 加载以bin格式储存的model
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.model = model

    def jieba_cut_sentences(self,in_filename,out_filename):
        # 一个简单而单纯的jieba分词。没有使用停词和自定义词库，目的是最大程度保留切出来句子的完整性，提高速度
        with open(in_filename,'r') as fi,open(out_filename,'w+') as fo:
            in_file = fi.readlines()
            counter = 0
            for line in in_file:
                line = re.sub(r'[\r\n\t\d?\!\,\.\:\"\'\|\\：！，、； 。“”‘’？~"]', '', line).strip()
                line = ' '.join(jieba.cut(line))
                if counter % 1000 == 0: print("%d lines processed."%counter)
                counter +=1
                fo.write(line+'\n')
        print("Finished!")

    def init_model(self,model_path='../models/beibei_gensim.model',corpus_path="../source/jieba_cut_pure_ques.txt",size=200):
        # 初始化一个word2vec模型，使用的语料为corpus_path中的语料
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence(corpus_path)  # 加载语料
        model = word2vec.Word2Vec(sentences, size=size, iter=10)  # 默认window=5
        model.save(model_path)
        print('Model is successfully saved!')

    def test_model(self,model_path='../models/beibei_gensim.model'):
        # 加载模型
        # self.load_model(model_path)
        model = self.model
        # 计算两个词的相似度/相关程度
        y1 = model.similarity(u"尺码", u"号码")
        print(u"【尺码】和【尺寸】的相似度为：", y1)
        print("--------\n")

        # 计算某个词的相关词列表
        y2 = model.most_similar(u"尺寸", topn=20)  # 20个最相关的
        print(u"和【尺寸】最相关的词有：\n")
        for item in y2:
            print(item[0], item[1])
        print("--------\n")

        # 寻找对应关系
        print(u"号码-尺码，大小-")
        y3 = model.most_similar([u'号码', u'尺码'], [u'大小'], topn=3)
        for item in y3:
            print(item[0], item[1])
        print("--------\n")

        # 寻找不合群的词
        y4 = model.doesnt_match(u"时候 1时间 明天 几点 订单".split())
        print(u"不合群的词：", y4)
        print("--------\n")

    def train_model(self,model_path='../models/beibei_gensim.model',corpus_path="../source/jieba_cut_all_test.txt"):
        # 这个函数只针对格式为.model的模型有效，作用是在已经经过训练的模型基础上继续训练
        model = self.model
        sentences = word2vec.LineSentence(corpus_path)
        model.train(sentences=sentences, total_examples=model.corpus_count,epochs=model.iter)
        model.save(model_path)
        print('Model is successfully trained and saved!')

if __name__ == '__main__':
    test = Accuracy_prediction('../models/word2vec_version1.model', 200)
    test.integration_testing()