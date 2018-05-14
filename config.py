# coding: utf-8
# Author: Miracle Yoo
import torch

class Config(object):
    def __init__(self):
        # 公共参数设置
        self.USE_CUDA           = torch.cuda.is_available()
        self.RUNNING_ON_SERVER  = False
        self.NET_SAVE_PATH      = "./source/trained_net/"
        self.TRAIN_DATASET_PATH = "../database/test_train/knowledge&log_data.txt"
        self.TEST_DATASET_PATH  = "../database/test_train/yibot_two_year_test.txt"
        self.NUM_EPOCHS         = 1000
        self.BATCH_SIZE         = 8
        self.TOP_NUM            = 3
        self.NUM_WORKERS        = 1
        self.IS_TRAINING        = True
        self.ENSEMBLE_TEST      = False
        self.LEARNING_RATE      = 0.001 
        self.RE_TRAIN           = False
        self.TEST_POSITION      = 'Gangge Server'

        # 模型共享参数设置
        self.OPTIMIZER          = 'Adam'
        self.USE_CHAR           = True
        self.USE_WORD2VEC       = True
        self.NUM_CLASSES        = 1890
        self.EMBEDDING_DIM      = 512
        self.VOCAB_SIZE         = 20029
        self.CHAR_SIZE          = 3403

        # LSTM模型设置
        self.LSTM_HID_SIZE      = 512
        self.LSTM_LAYER_NUM     = 2
        
        # TextCNN模型设置
        self.TITLE_DIM          = 200
        self.SENT_LEN           = 20
        self.LINER_HID_SIZE     = 2000
        self.KERNEL_SIZE        = [1,2,3,4,5]

        # DilaTextCNN模型设置
        self.DILA_TITLE_DIM     = 20

        


