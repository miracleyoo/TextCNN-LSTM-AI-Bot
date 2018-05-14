# coding: utf-8
# Author: Miracle Yoo
from torch.utils.data import DataLoader
from utils import *
from train import *
from config import Config
from LoadData import *
from models import LSTM, BiLSTM, TextCNN, OriTextCNN

opt = Config()

folder_init()
prep = PrepareData(char=True)
if type(opt.TRAIN_DATASET_PATH)  == list:
    data_dict    = prep.gen_data_dict(data_list=multi_dataset_merge(*opt.TRAIN_DATASET_PATH))
else:
    data_dict    = prep.gen_data_dict(opt.TRAIN_DATASET_PATH)
test_dict    = prep.gen_data_dict(opt.TEST_DATASET_PATH)

if opt.IS_TRAINING:
    try:
        vocab_dict = pickle.load(open("./source/data/vocab_dict.pkl", "rb"))
        title      = pickle.load(open("./source/data/title.pkl", "rb"))
    except:
        vocab_dict = prep.gen_vocab_dict(data_dict,test_dict)
        title      = prep.gen_title(data_dict, test_dict)
        pickle.dump(vocab_dict, open("./source/data/vocab_dict.pkl", "wb"))
        pickle.dump(title, open("./source/data/title.pkl", "wb"))
else:
    vocab_dict     = pickle.load(open("./source/data/vocab_dict.pkl", "rb"))
    title          = pickle.load(open("./source/data/title.pkl", "rb"))

if opt.USE_CHAR: opt.CHAR_SIZE = len(vocab_dict)
else: opt.VOCAB_SIZE = len(vocab_dict)
opt.NUM_CLASSES    = len(title)
# trainData    = prep.load_cls_data(data_dict, title, train=True)
testData     = prep.load_cls_data(test_dict, title, train=False)
trainDataSet = BalancedData(data_dict,title,opt,vocab_dict=vocab_dict)
# trainDataSet = BeibeiClassification(trainData[:32], vocab_dict=vocab_dict, char=True)
testDataSet  = BeibeiClassification(testData[:] , vocab_dict=vocab_dict, char=True)
train_loader = DataLoader(dataset=trainDataSet, batch_size=opt.BATCH_SIZE,
                shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=True)

test_loader  = DataLoader(dataset=testDataSet, batch_size=opt.BATCH_SIZE,
                shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=True)

if opt.IS_TRAINING:
    net = training(train_loader, test_loader, opt)
else:
    if opt.ENSEMBLE_TEST:
        net_list = [OriTextCNN.OriTextCNN(opt),BiLSTM.BiLSTM(opt)]
        for i, _ in enumerate(net_list):
            net_list[i], *_  = net_list[i].load(opt.NET_SAVE_PATH+ net_list[i].model_name + "/best_model.dat")
        ave_test_loss, ave_test_acc, _ = ensemble_testing(test_loader, net_list, opt)
    else:
        net      = LSTM.LSTM(opt)
        net, *_   = net.load(opt.NET_SAVE_PATH + net.model_name + "/best_model.dat")
        print('==> Now testing model: %s '%(net.model_name))
        ave_test_loss, ave_test_acc = testing(test_loader, net, opt)

    print( 'Test Loss: %.4f, Test Acc: %.4f'%(ave_test_loss, ave_test_acc))
