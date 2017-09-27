#!/usr/bin/python
#encoding=utf-8

from data.data_loader import *
from model import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import csv

USE_CUDA=True

def test(model_path):
    package = torch.load(model_path)

    net1 = package['net1']
    print(net1)
    net2 = package['net2']
    print(net2)
    layers = package['layers']
    dropout = package['dropout']
    bias = package['bias']
    batch_size = 256

    model = Model(net1, net2, dropout=dropout, bias=bias)
    '''
    rnn_input_size = package['rnn_input_size']
    rnn_hidden_size = package['rnn_hidden_size']
    rnn_layers = package['rnn_layers']
    rnn_type = package['rnn_type']
    num_class = package['num_class']
    dropout = package['dropout']
    batch_size=64

    model = RNNModel(rnn_input_size=rnn_input_size, rnn_hidden_size=rnn_hidden_size, 
                            rnn_layers=rnn_layers, rnn_type=rnn_type, bidirectional=True,
                            batch_norm=True, num_class=num_class, dropout=dropout)
    '''
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    test_dataset = myDataset(date='20170923', data_set='test', n_feats=88, window=None)
    test_loader = myDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=False)
    
    result = []
    for data in test_loader:
        inputs, ids = data
        inputs = Variable(inputs, volatile=True, requires_grad=False)
        if USE_CUDA:
            inputs = inputs.cuda()
        
        probs = model(inputs)
        #batch_size, window, dim = probs.size()
        #probs = probs.view(-1, dim)
        
        pred = torch.exp(probs)
        #pred = torch.exp(probs)
        #for i in range(len(ids)):
        #    for j in range(window):
        #        if ids[i][j] is not None:
        #            result.append([int(ids[i][j]), pred[i*window+j][1].data[0]])
        for i in range(len(ids)):
            result.append([ids[i], pred[i][1].data[0]])
    #result.sort(key=lambda x:x[0])
    #print(result)
    f = open('./result/test_csv_20170927_1.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['id', 'proba'])
    writer.writerows(result)
    f.close()
    print("End predict!")

if __name__ == "__main__":
    test('./log/best_model_cv0.727206894467.pkl')


