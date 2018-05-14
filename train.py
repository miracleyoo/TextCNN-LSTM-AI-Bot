# coding: utf-8
# Author: Miracle Yoo
import torch
import torch.nn as nn
import torch.autograd
import json

from torch.autograd import Variable
from utils import *
from tqdm import tqdm
from models import LSTM, BiLSTM, DilaTextCNN, OriTextCNN, TextCNNInc


def training(train_loader, test_loader, opt):
    net_list = [LSTM.LSTM(opt), BiLSTM.BiLSTM(opt), OriTextCNN.OriTextCNN(opt),
                DilaTextCNN.DilaTextCNN(opt), TextCNNInc.TextCNNInc(opt)]
    net = net_list[4]
    best_acc = 0
    # best_top_accs   = []
    best_acc_loss = 0
    NUM_TRAIN = len(train_loader) * opt.BATCH_SIZE
    PRE_EPOCH = 0
    NUM_TRAIN_PER_EPOCH = len(train_loader)
    NET_PREFIX = opt.NET_SAVE_PATH + net.model_name + "/"
    print('==> Loading Model ...')
    temp_model_name = "temp_model.dat"
    model_name = "best_model.dat"
    if not os.path.exists(NET_PREFIX):
        os.mkdir(NET_PREFIX)
    if not os.path.exists('./source/log/' + net.model_name):
        os.mkdir('./source/log/' + net.model_name)
    if os.path.exists(NET_PREFIX + temp_model_name) and opt.RE_TRAIN == False:
        try:
            net, PRE_EPOCH, best_acc = net.load(NET_PREFIX + temp_model_name)
            print("Load existing model: %s" % (NET_PREFIX + temp_model_name))
        except:
            pass

    if opt.USE_CUDA: net.cuda()

    criterion = nn.CrossEntropyLoss()
    if opt.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)
    elif opt.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.LEARNING_RATE)
    elif opt.OPTIMIZER == 'RMSP':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.LEARNING_RATE)
    else:
        raise NameError("This optimizer isn't defined")

    train_recorder = {'loss': [], 'acc': []}
    test_recorder = {'loss': [], 'acc': []}

    t = datetime.datetime.now().strftime("%m%d_%H:%M:%S")
    log_file_name = "%s_%s.txt" % (net.model_name, t)

    for epoch in range(opt.NUM_EPOCHS):
        train_loss = 0
        train_acc = 0

        # Start training
        net.train()
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=NUM_TRAIN_PER_EPOCH, leave=False, unit='b'):
            inputs, labels, sent = data
            if opt.USE_CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Do statistics for training
            train_loss += loss.data[0]
            _, predicts = torch.max(outputs, 1)
            num_correct = (predicts == labels).sum()
            train_acc += num_correct.data[0]

        # Start testing
        ave_test_loss, ave_test_acc, topnacc = testing(test_loader, net, opt)
        ave_train_loss = float(train_loss) / NUM_TRAIN
        ave_train_acc = float(train_acc) / NUM_TRAIN

        # Do recording for each epoch
        train_recorder['loss'].append(ave_train_loss)
        train_recorder['acc'].append(ave_train_acc)
        test_recorder['loss'].append(ave_test_loss)
        test_recorder['acc'].append(ave_test_acc)

        # Write log to files
        with open('./source/log/' + net.model_name + '/' + log_file_name, 'w+') as fp:
            json.dump({'train_recorder': train_recorder, 'test_recorder': test_recorder}, fp)

        # Output results
        print('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f'
              % (epoch + 1 + PRE_EPOCH, opt.NUM_EPOCHS + PRE_EPOCH,
                 ave_train_loss, ave_train_acc,
                 ave_test_loss, ave_test_acc))
        if ave_test_acc > best_acc:
            best_acc = ave_test_acc
            best_top_accs = topnacc
            best_acc_loss = ave_test_loss
            net.save((epoch + PRE_EPOCH), best_acc, model_name)

        # Save a temp model
        net.save((epoch + PRE_EPOCH), best_acc, temp_model_name)

    summary_info = {'total_epoch': (epoch + PRE_EPOCH), 'best_acc': best_acc,
                    'best_acc_loss': best_acc_loss, 'ave_test_acc': ave_test_acc,
                    'ave_test_loss': ave_test_loss, 'ave_train_acc': ave_train_acc,
                    'ave_train_loss': ave_train_loss, 'best_top_accs': best_top_accs}
    write_summary(net, opt, summary_info)
    print('==> Training Finished. Current model is %s. The highest test acc is %.4f' % (net.model_name, best_acc))
    return net


def testing(test_loader, net, opt):
    NUM_TEST_PER_EPOCH = len(test_loader)
    NUM_TEST = NUM_TEST_PER_EPOCH * opt.BATCH_SIZE
    test_loss = 0
    test_acc = 0
    topn_acc = [0] * opt.TOP_NUM
    criterion = nn.CrossEntropyLoss()
    if opt.USE_CUDA: net.cuda()

    net.eval()
    for i, data in tqdm(enumerate(test_loader), desc="Testing", total=NUM_TEST_PER_EPOCH, leave=False, unit='b'):
        inputs, labels, sent = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.data[0]
        _, predicts = torch.max(outputs, 1)
        num_correct = (predicts == labels).sum()

        for i in range(opt.TOP_NUM):
            predictsn = np.array(outputs.data.sort(descending=True, dim=1)[1])[:, :opt.TOP_NUM]
            # print(predicts.shape,labels)
            for j in range(len(labels)):
                if labels.data[j] in predictsn:
                    topn_acc[i] += 1

        # Do statistics for training
        test_loss += loss.data[0]
        test_acc += num_correct.data[0]

    test_loss = float(test_loss)
    test_acc = float(test_acc)
    topn_acc = [float(x) / NUM_TEST for x in topn_acc]
    return test_loss / NUM_TEST, test_acc / NUM_TEST, topn_acc


def ensemble_testing(test_loader, net_list, opt):
    NUM_TEST_PER_EPOCH = len(test_loader)
    NUM_TEST = NUM_TEST_PER_EPOCH * opt.BATCH_SIZE
    test_loss = 0
    test_acc = 0
    criterion = nn.CrossEntropyLoss()
    if opt.USE_CUDA: net_list = [net.cuda() for net in net_list]
    # print(net_list)
    for i, _ in enumerate(net_list):
        net_list[i].eval()
    for i, data in tqdm(enumerate(test_loader), desc="Testing", total=NUM_TEST_PER_EPOCH, leave=False, unit='b'):
        inputs, labels, sent = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        if opt.USE_CUDA:
            outputs = Variable(torch.zeros(opt.BATCH_SIZE, opt.NUM_CLASSES)).cuda()
        else:
            outputs = Variable(torch.zeros(opt.BATCH_SIZE, opt.NUM_CLASSES))
        for net in net_list:
            outputs += net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.data[0]
        _, predicts = torch.max(outputs, 1)
        num_correct = (predicts == labels).sum()

        # Do statistics for training
        test_loss += loss.data[0]
        test_acc += num_correct.data[0]

    test_loss = float(test_loss)
    test_acc = float(test_acc)
    return test_loss / (NUM_TEST * len(net_list)), test_acc / NUM_TEST
