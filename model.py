#!/usr/bin/python
#encoding=utf-8

import torch
import torch.nn as nn
from collections import OrderedDict
import copy

class Model(nn.Module):
    def __init__(self, net=[88, 1024, 2], dropout=0.5, bias=True):
        super(Model, self).__init__()
        self.layers = len(net)
        self.net = copy.deepcopy(net)
        self.dropoutp = dropout
        self.bias = bias
        nns = []
        self.dropout = nn.Dropout(dropout)
        #self.conv = nn.Conv1d(1,1,3)
        #net[0] = (net[0] - 2)
        for i in range(self.layers-2):
            linear = nn.Linear(net[i], net[i+1], bias=True)
            nns.append(('Linear%d' % (i+1), linear))
            nns.append(('RELU%d' % (i+1), nn.ReLU()))
            nns.append(('BatchNorm%d' % (i+1), nn.BatchNorm1d(net[i+1])))
        self.linear = nn.Sequential(OrderedDict(nns))
        self.OutLayer = nn.Linear(net[-2], net[-1], bias = bias)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        #sizes = inputs.size()
        #inputs = self.conv(inputs.view(sizes[0],1,-1))
        inputs = self.dropout(inputs)
        inputs = self.linear(inputs)
        inputs = self.OutLayer(inputs)
        inputs = self.softmax(inputs)

        return inputs
    
    @staticmethod
    def save_package(model, optimizer=None, epoch=None, loss_results=None, training_cer_results=None, dev_cer_results=None):
        package = {
                'net': model.net,
                'dropout': model.dropoutp,
                'layers': model.layers,
                'bias': model.bias,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['train_loss_results'] = loss_results
            package['dev_loss_results'] = training_cer_results
            package['dev_acc_results'] = dev_cer_results
        return package

class RNNModel(nn.Module):
    def __init__(self, rnn_input_size=88, hidden_size=256, rnn_type=nn.LSTM, bidirectinak=False, 
                        batch_norm=True, num_class=2, dropout=0.5):
        super(RNNModel, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.net = copy.deepcopy(net)
        self.dropoutp = dropout
        self.bias = bias
        nns = []
        self.dropout = nn.Dropout(dropout)
        #self.conv = nn.Conv1d(1,1,3)
        #net[0] = (net[0] - 2)
        for i in range(self.layers-2):
            linear = nn.Linear(net[i], net[i+1], bias=True)
            nns.append(('Linear%d' % (i+1), linear))
            nns.append(('RELU%d' % (i+1), nn.ReLU()))
            nns.append(('BatchNorm%d' % (i+1), nn.BatchNorm1d(net[i+1])))
        self.linear = nn.Sequential(OrderedDict(nns))
        self.OutLayer = nn.Linear(net[-2], net[-1], bias = bias)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        #sizes = inputs.size()
        #inputs = self.conv(inputs.view(sizes[0],1,-1))
        inputs = self.dropout(inputs)
        inputs = self.linear(inputs)
        inputs = self.OutLayer(inputs)
        inputs = self.softmax(inputs)

        return inputs
    
    @staticmethod
    def save_package(model, optimizer=None, epoch=None, loss_results=None, training_cer_results=None, dev_cer_results=None):
        package = {
                'net': model.net,
                'dropout': model.dropoutp,
                'layers': model.layers,
                'bias': model.bias,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['train_loss_results'] = loss_results
            package['dev_loss_results'] = training_cer_results
            package['dev_acc_results'] = dev_cer_results
        return package 
