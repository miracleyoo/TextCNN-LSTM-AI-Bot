# coding: utf-8
# Author: Miracle Yoo
import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np
from .BasicModule import BasicModule
torch.manual_seed(233)
np.random.seed(233)
"""
Neural Networks model : LSTM
"""

class BiLSTM(BasicModule):
    
    def __init__(self, opt):
        super(BiLSTM, self).__init__()
        self.hidden_dim       = opt.LSTM_HID_SIZE
        self.batch_size       = opt.BATCH_SIZE
        self.number_layers    = opt.LSTM_LAYER_NUM
        self.embedding_dim    = opt.EMBEDDING_DIM
        self.output_dim       = opt.NUM_CLASSES
        self.use_cuda         = opt.USE_CUDA
        if opt.USE_CHAR: self.encoder = nn.Embedding(opt.CHAR_SIZE, opt.EMBEDDING_DIM)
        else: self.encoder    = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        self.encoder          = nn.Embedding(opt.CHAR_SIZE, opt.EMBEDDING_DIM)
        self.lstm             = nn.LSTM(self.embedding_dim, self.hidden_dim, 
            self.number_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc               = nn.Linear(2*self.hidden_dim, self.output_dim)
        self.hidden           = self.init_hidden()
        
    def init_hidden(self):
        if self.use_cuda:
            return (Variable(torch.zeros(2*self.number_layers, self.batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(2*self.number_layers, self.batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(2*self.number_layers, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2*self.number_layers, self.batch_size, self.hidden_dim)))
        
    def forward(self, x):
        lstm_out, self.hidden = self.lstm(self.encoder(x), self.hidden)
        output                = lstm_out[:,-1,:]
        output                = self.fc(output)
        self.hidden           = self.init_hidden()
        return output

# net = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(main_ques))
# print(net)