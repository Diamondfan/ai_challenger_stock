#!/usr/bin/python
#encoding=utf-8

import os
import h5py
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv

data_dir = '/home/fan/AI_challenger/stock/data/'
max_group = 28

#Override the class of Dataset
class myDataset(Dataset):
    def __init__(self, date, data_set='train', n_feats=88, window=None):
        self.data_set = data_set
        self.n_feats = n_feats
        self.window = window

        h5_file = os.path.join(data_dir, date, data_set+'.h5py')
        if not os.path.exists(h5_file):
            print("File not exist. Please process %s data in csv format !" % data_set)
        else:
            print("Loading %s data from h5py file..." % data_set)
            self.load_h5py(h5_file)

    def load_h5py(self, h5_file):
        self.features_label = []
        f = h5py.File(h5_file, 'r')
        if self.window == None:
            self.samples = np.asarray(f['data'])
        elif self.data_set != 'test':
            samples = np.asarray(f['data'])
            self.samples = []
            self.weights = []
            self.labels = []
            n = 0
            group = -1
            for i in range(len(samples)):
                if int(samples[i][-1]) == group:
                    if n % self.window == 0:
                        self.samples.append(np.array(sample))
                        self.labels.append(np.array(label))
                        self.weights.append(np.array(weight))
                        sample = []
                        label = []
                        weight = []
                    sample.append(samples[i][:self.n_feats])
                    label.append(samples[i][-2])
                    weight.append(samples[i][-3])
                else:
                    #Todo:  把省略的样本加上
                    n = 0
                    group = int(samples[i][-1])
                    sample = [samples[i][:self.n_feats]]
                    label = [samples[i][-2]]
                    weight = [samples[i][-3]]
                n += 1
        else:
            samples = np.asarray(f['data'])
            self.samples = []
            self.ids = []
            n = 0
            group = -1
            for i in range(len(samples)):
                if int(samples[i][-1]) == group:
                    if n % self.window == 0:
                        self.samples.append(np.array(sample))
                        self.ids.append(np.array(idw))
                        sample = []
                        idw = []
                    sample.append(samples[i][:self.n_feats])
                    idw.append(samples[i][-2])
                else:
                    if n!=0:
                        if n % self.window == 0:
                            pad_num = 0
                        else:
                            pad_num = (self.window - n % self.window)
                        for pad in range(pad_num):
                            sample.append([0]*88)
                            idw.append(None)
                        self.samples.append(np.array(sample))
                        self.ids.append(np.array(idw))
                    n = 0
                    group = int(samples[i][-1])
                    sample = [samples[i][:self.n_feats]]
                    idw = [samples[i][-2]]
                n += 1
            if n % self.window == 0:
                pad_num = 0
            else:
                pad_num = (self.window - n % self.window)
            for pad in range(pad_num):
                sample.append([0]*88)
                idw.append(None)
            self.samples.append(np.array(sample))
            self.ids.append(np.array(idw))
                 
        print("Load %d samples from %s dataset" % (self.__len__(), self.data_set))

    def __getitem__(self, idx):
        if self.window == None:
            return self.samples[idx]
        elif self.data_set != 'test':
            return (self.samples[idx], self.labels[idx], self.weights[idx])
        else:
            return (self.samples[idx], self.ids[idx])

    def __len__(self):
        return len(self.samples) 

#前馈神经网络输入处理
def create_input(batch):  
    batch_size = len(batch)
    inputs = torch.zeros(batch_size, 88)
    if len(batch[0]) == 91:
        labels = []
        weights = []
        for x in range(batch_size):
            sample = batch[x]
            #group = int(sample[-1])
            label = int(sample[-2])
            weight = int(sample[-3])
            feature = torch.FloatTensor(sample[:-3])
            #start = (group-1)*88
            inputs[x].copy_(feature)
            labels.append(label)
            weights.append(weight)
        labels = torch.LongTensor(labels)
        weights = torch.FloatTensor(weights)
        return inputs, weights, labels
    else:
        ids = []
        for x in range(batch_size):
            sample = batch[x]
            group = int(sample[-1])
            ids.append(int(sample[-2]))
            feature = torch.FloatTensor(sample[:-2])
            inputs[x].copy_(feature)
        return inputs, ids

#rnn输入处理
def sequence_input(batch):  
    batch_size = len(batch)
    window, feat_size = batch[0][0].shape
    inputs = torch.zeros(batch_size, window, feat_size)
    if  len(batch[0]) == 3:
        labels = []
        weights = []
        for x in range(batch_size):
            sample, label, weight = batch[x]
            feature = torch.FloatTensor(sample)
            inputs[x].copy_(feature)
            labels.append(label)
            weights.append(weight)
        labels = torch.LongTensor(np.array(labels))
        weights = torch.FloatTensor(np.array(weights))
        return inputs, weights, labels
    else:
        ids =[]
        for x in range(batch_size):
            sample, idw = batch[x]
            ids.append(idw)
            feature = torch.FloatTensor(sample)
            inputs[x].copy_(feature)
        return inputs, ids

#class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
#                           sampler=None, batch_sampler=None, num_workers=0, 
#                           collate_fn=<function default_collate>, 
#                           pin_memory=False, drop_last=False)
#subclass of DataLoader and rewrite the collate_fn to form batch

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        if self.dataset.window == None:
            self.collate_fn = create_input
        else:
            self.collate_fn = sequence_input

if __name__ == '__main__':
    dev_dataset = myDataset(date='20170916/sequence', data_set='train', n_feats=88, window=None)
    #for i in range(len(dev_dataset)):
    #    if 501554 in dev_dataset[i][1]:
    #        print('yes')
    #print('no')
    #print(len(dev_dataset))
    dev_loader = myDataLoader(dev_dataset, batch_size=4, shuffle=True, 
                     num_workers=4, pin_memory=False)
    #i = 0
    #for data in dev_loader:
    #    if i == 0:
    #        print(data)
    #    i += 1

