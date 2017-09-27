#!/usr/bin/python
#encoding=utf-8

import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, net1=[88, 128, 88], net2=[16,4,2], dropout=0.5, bias=True):
        super(Model, self).__init__()
        self.layers = len(net1)+len(net2)
        self.net1 = copy.deepcopy(net1)
        self.net2 = copy.deepcopy(net2)
        self.dropoutp = dropout
        self.bias = bias
        
        nns = []
        for i in range(len(net1)-1):
            linear = nn.Linear(net1[i], net1[i+1], bias=True)
            nns.append(('net1, Linear%d'%(i+1), linear))
            nns.append(('net1, ReLu%d'%(i+1), nn.ReLU()))
            nns.append(('net1, Dropout%d'%(i+1), nn.Dropout(dropout)))
        self.transform = nn.Sequential(OrderedDict(nns))

        self.Linear = nn.Linear(net1[-1], net2[0], bias=bias)
        
        nns = []
        for i in range(len(net2)-2):
            linear = nn.Linear(net2[i], net2[i+1], bias=True)
            nns.append(('Linear%d' % (i+1), linear))
            nns.append(('RELU%d' % (i+1), nn.ReLU()))
            nns.append(('BatchNorm%d' % (i+1), nn.BatchNorm1d(net2[i+1])))
        self.linear = nn.Sequential(OrderedDict(nns))
        
        self.OutLayer = nn.Linear(net2[-2], net2[-1], bias = bias)
        
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        #sizes = inputs.size()
        #inputs = self.conv(inputs.view(sizes[0],1,-1))
        inputs = self.transform(inputs)
        inputs = self.Linear(inputs)
        inputs = self.linear(inputs)
        inputs = self.OutLayer(inputs)
        inputs = self.softmax(inputs)

        return inputs
    
    @staticmethod
    def save_package(model, optimizer=None, epoch=None, train_loss_results=None, dev_loss_results=None, dev_acc_results=None):
        package = {
                'net1': model.net1,
                'net2': model.net2,
                'dropout': model.dropoutp,
                'layers': model.layers,
                'bias': model.bias,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if epoch is not None:
            package['epoch'] = epoch
        if train_loss_results is not None:
            package['train_loss_results'] = train_loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_acc_results'] = dev_acc_results
        return package

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module
    
    def forward(self, x):
        batch_size, window, feat_size = x.size()
        x = x.view(-1, feat_size)
        x = self.module(x)
        x = x.view(batch_size, window, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class InferenceBatchLogSoftmax(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        return torch.stack([F.log_softmax(x[i]) for i in range(batch_size)], 0)

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                    bidirectional=False, batch_norm=True, dropout = 0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                                bidirectional=bidirectional, dropout = dropout, bias=False)
    
    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        self.rnn.flatten_parameters()
        return x

class RNNModel(nn.Module):
    def __init__(self, rnn_input_size=88, rnn_hidden_size=256, rnn_layers=4, rnn_type=nn.LSTM, 
                        bidirectional=False, batch_norm=True, num_class=2, dropout=0.1):
        super(RNNModel, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.num_class = num_class
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        
        rnns = []
        rnn = BatchRNN(input_size = rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                        bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers - 1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, hidden_size=rnn_hidden_size, 
                            rnn_type = rnn_type, bidirectional=bidirectional, 
                            dropout = dropout, batch_norm=batch_norm)
            rnns.append(('%d' % (i+1), rnn))
        
        self.rnns = nn.Sequential(OrderedDict(rnns))
        
        if batch_norm:
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions*rnn_hidden_size),
                                nn.Linear(self.num_directions*rnn_hidden_size, num_class, bias=False))
        else:
            fc = nn.Linear(self.num_directions*rnn_hidden_size, num_class, bias=False)
        
        self.fully_connect = SequenceWise(fc)
        self.softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        x = self.rnns(x)
        
        x = self.fully_connect(x)
        
        x = self.softmax(x)
        return x
    
    @staticmethod
    def save_package(model, optimizer=None, epoch=None, train_loss_results=None, dev_loss_results=None, dev_acc_results=None):
        package = {
                'rnn_input_size': model.rnn_input_size,
                'rnn_hidden_size': model.rnn_hidden_size,
                'rnn_layers': model.rnn_layers,
                'rnn_type': model.rnn_type,
                'num_class':model.num_class,
                'dropout': model.dropout,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if epoch is not None:
            package['epoch'] = epoch
        if train_loss_results is not None:
            package['train_loss_results'] = train_loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_acc_results'] = dev_acc_results
        return package 
