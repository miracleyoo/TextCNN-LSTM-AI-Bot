# coding: utf-8
# Author: Miracle Yoo

from .BasicModule import BasicModule
import jieba
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gensim.models import word2vec
torch.manual_seed(1)

class DilaTextCNN(BasicModule):
    def __init__(self, opt):
        super(DilaTextCNN, self).__init__()
        self.model_name    = "DilaTextCNN"
        if opt.USE_CHAR:
            opt.VOCAB_SIZE = opt.CHAR_SIZE
        self.encoder       = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        question_convs     = []
        for i in range(1,6):
            for j in range(1,4):
                if j==1:
                    dd_list = [0]
                else:
                    max_dd  = (opt.SENT_LEN-i-j+1)//(j-1)
                    dd_list = range(min((max_dd+1),4)) 
                for k in dd_list:
                    med_dim = (200-10*(i+j))//(5*k+1)
                    # print(i,j,k,med_dim)
                    temp_seq = nn.Sequential(
                        nn.Conv1d(in_channels  =opt.EMBEDDING_DIM,
                                  out_channels =med_dim,
                                  kernel_size  =i),
                        nn.BatchNorm1d(med_dim),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv1d(in_channels  =med_dim,
                                  out_channels =med_dim,
                                  kernel_size  =j,
                                  dilation     =(k+1)),
                        nn.BatchNorm1d(med_dim),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=(opt.SENT_LEN -i-(j-1)*(k+1)+1))
                        )
                    question_convs.append(temp_seq)

        self.question_convs = nn.ModuleList(question_convs)
        self.fc = nn.Sequential(
            nn.Linear(2700, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASSES)
        )
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, question):
        question = self.encoder(question)
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        out      = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        out      = torch.cat(out, dim=1)
        print(out.size())
        out      = out.permute(0, 2, 1)

        out      = out.view(out.size(0), -1)
        logits   = self.fc(out)
        return logits


