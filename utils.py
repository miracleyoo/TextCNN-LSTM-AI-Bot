# coding: utf-8
# Author: Miracle Yoo
import codecs
import os
import datetime
import numpy as np

import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")


def folder_init():
    """
    Initialize folders required
    """
    if not os.path.exists('source'):os.mkdir('source')
    if not os.path.exists('source/log'):os.mkdir('source/log')
    if not os.path.exists('source/trained_net'):os.mkdir('source/trained_net')
    if not os.path.exists('source/data'): os.mkdir('source/data')
    if not os.path.exists('source/nohup'): os.mkdir('source/nohup')
    if not os.path.exists('source/summaries'): os.mkdir('source/summaries')


def change_dataset_format(filename):
    """
    Change the csv dataset file which fanyu using to stadard || dataset file.
    """
    oridf = pd.read_csv(filename)
    data  = []
    for line in oridf.iterrows():
        data.append(line[1]['query']+'||'+line[1]['main_question']+'\n')
    with open(os.path.splitext(filename)[0]+'.txt','w+') as f:
        f.writelines(data)


def multi_dataset_merge(*filenames):
    """
    :param filenames: multiple dataset path
    :return: merged dataset list
    """
    data = []
    for filename in filenames:
        with open(filename) as f:
            data.extend(f.readlines())
    return(data)


def get_topn_acc(outputs, labels, top_num=3):
    # print(labels,outputs.size())
    acc = [0] * top_num
    for i in range(top_num):
        predicts = np.array(outputs.sort(descending=True, dim=1)[1])[:,:top_num]
        # print(predicts.shape,labels)
        for j in range(len(labels)):
            if labels[j] in predicts:
                acc[i]+=1
    acc = [x/len(labels) for x in acc]
    return acc


def write_summary(net, opt, summary_info):
    print(summary_info['best_top_accs'])
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix   = './source/summaries/'+net.model_name
    if not os.path.exists(prefix): os.mkdir(prefix)
    sum_path = prefix + '/MiracleYoo_'+current_time+'_'+net.model_name+'_模型测试记录表.md'
    with codecs.open('./config.py', 'r', encoding='utf-8') as f:
        raw_data = f.readlines()
        configs  = ''
        for line in raw_data:
            if line.strip().startswith('self.'):
                configs += line.strip().lstrip('self.')+'\n'

    content = '''
# Model Testing Record Form
| Item Name        | Information |
| ---------        | ----------- |
| Model Name       | %s          |
| Tester's Name    | Miracle Yoo |
| Author's Nmae    | Miracle Yoo |
| Test Time        | %s          |
| Test Position    | %s          |
| Training Epoch   | %d          |
| Highest Test Acc | %.4f        |
| Loss of best Test Acc | %.4f   |
| Top2Acc of best Test Acc|%.4f  |
| Top3Acc of best Test Acc|%.4f  |
| Last epoch test acc   | %.4f   |
| Last epoch test loss  | %.4f   |
| Last epoch train acc  | %.4f   |
| Last epoch train loss | %.4f   |
| Train Dataset Path    | %s     |
| Test Dataset Path     | %s     |
| Class Number     | %d          |
| Framwork         | Pytorch     |
| Basic Method     | Classify    |
| Input Type       | Char        |
| Criterion        | CrossEntropy|
| Optimizer        | %s          |
| Learning Rate    | %.4f        |
| Embedding dimension   | %d     |
| Data Homogenization   | True   |
| Pretreatment|Remove punctuation|
| Other Major Param |            |
| Other Operation   |            |


## Configs
```
%s
```

## Net Structure
```
%s
```
    '''%(
        net.model_name,
        current_time,
        opt.TEST_POSITION,
        summary_info['total_epoch'],
        summary_info['best_acc'],
        summary_info['best_acc_loss'],
        summary_info['best_top_accs'][1],
        summary_info['best_top_accs'][2],
        summary_info['ave_test_acc'],
        summary_info['ave_test_loss'],
        summary_info['ave_train_acc'],
        summary_info['ave_train_loss'],
        os.path.basename(opt.TRAIN_DATASET_PATH),
        os.path.basename(opt.TEST_DATASET_PATH),
        opt.NUM_CLASSES,
        opt.OPTIMIZER,
        opt.LEARNING_RATE,
        opt.EMBEDDING_DIM,

        configs.strip('\n'),
        str(net)
    )
    with codecs.open(sum_path, 'w+', encoding='utf-8') as f:
        f.writelines(content)