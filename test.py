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

    net = package['net']
    print(net)
    layers = package['layers']
    dropout = package['dropout']
    bias = package['bias']
    batch_size = 256

    model = Model(net, dropout=dropout, bias=bias)
    
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    test_dataset = myDataset(date='20170916/min_max', data_set='test', n_feats=88)
    test_loader = myDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=False)
    
    result = []
    for data in test_loader:
        inputs, ids = data
        inputs = Variable(inputs, volatile=True, requires_grad=False)
        if USE_CUDA:
            inputs = inputs.cuda()
        
        probs = model(inputs)
        pred = torch.exp(probs)
        print(pred)
        #pred = torch.exp(probs)
        for i in range(len(ids)):
            result.append([ids[i], pred[i][1].data[0]])
    #print(result)
    f = open('./result/test_csv_20170921_6.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['id', 'proba'])
    writer.writerows(result)
    f.close()
    print("End predict!")

if __name__ == "__main__":
    test('./log/best_model_cv0.68701003587.pkl')

