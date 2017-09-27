#!/usr/bin/python
#encoding=utf-8

import csv
import random
import h5py
import numpy as np

train_file = './20170923/ai_challenger_stock_train_20170923/stock_train_data_20170923.csv'
test_file = './20170923/ai_challenger_stock_test_20170923/stock_test_data_20170923.csv'
train_save_file = './20170923/train.h5py'
dev_save_file = './20170923/dev.h5py'
test_save_file = './20170923/test.h5py'

all_feature = []
#处理训练集
f = open(train_file, 'r')
reader = csv.reader(f)
#sortedlist = sorted(reader, key=lambda x:(x[-1], x[-2]))
sortedlist = reader
train_samples = []
dev_samples = []
train_feature = []
dev_feature = []
i = 0
pos = 0
neg = 0
for row in sortedlist:
    #if i == len(sortedlist)-1:
    #    break
    if i == 0:
        i += 1
        continue
    sample = []
    for idx in range(1,len(row)):
        sample.append(float(row[idx]))
    #if int(sample[-4]) == 0:
    #    continue
    #if int(sample[-2]) == 1:
    if int(sample[-1]) == 10:
        dev_feature.append(sample[:88])
        dev_samples.append(sample[88:-1])
    else:
        train_feature.append(sample[:88])
        train_samples.append(sample[88:-1])
    i += 1
    print("process:",i)
f.close()

#处理测试集
f = open(test_file, 'r')
reader = csv.reader(f)
#sortedlist = sorted(reader, key=lambda x:x[-1])
sortedlist = reader
i = 0
test_samples = []
test_feature = []
for row in sortedlist:
    #if i == len(sortedlist) -1:
    #    break
    if i == 0:
        i += 1
        continue
    sample = []
    for idx in range(len(row)):
        sample.append(float(row[idx]))
    test_samples.append([sample[0],sample[-1]])
    test_feature.append(sample[1:-1])
    i += 1
    print('process:',i)
f.close()

#数据归一化，减均值，处方差
train_feature = np.array(train_feature).astype(float)
dev_feature = np.array(dev_feature).astype(float)
test_feature = np.array(test_feature).astype(float)
'''
train_feature -= np.mean(train_feature, axis=0)
cov = np.dot(train_feature.T, train_feature) / train_feature.shape[0]
U, S, V = np.linalg.svd(cov)
train_feature = np.dot(train_feature, U)
dev_feature -= np.mean(dev_feature, axis=0)
cov = np.dot(dev_feature.T, dev_feature) / dev_feature.shape[0]
U, S, V = np.linalg.svd(cov)
dev_feature = np.dot(dev_feature, U)
'''
#x = np.rollaxis(all_feature, axis=0)
#pmax = np.max(all_feature, axis=0)
#pmin = np.min(all_feature, axis=0)
#x = (x-pmin)/(pmax-pmin)
#all_feature = x
'''
mean = np.mean(train_feature, axis=0)
std = np.std(train_feature, axis=0)
x = np.rollaxis(train_feature, axis=0)
x -= mean
x /= std
train_feature = x
x = np.rollaxis(np.array(dev_feature), axis=0)
x -= mean
x /= std
dev_feature = x
x = np.rollaxis(np.array(test_feature), axis=0)
x -= mean
x /= std
test_feature = x
'''

#分出验证集进行存储
train_samples = np.concatenate((train_feature, np.array(train_samples)), axis=1)
test_samples = np.concatenate((test_feature, np.array(test_samples)), axis=1)
dev_samples = np.concatenate((dev_feature, np.array(dev_samples)), axis=1)
print(train_samples.shape)
print(dev_samples.shape)
f1 = h5py.File(train_save_file, 'w')
f1.create_dataset('data', data=train_samples)
f2 = h5py.File(dev_save_file, 'w')
f2.create_dataset('data', data=np.array(dev_samples))
f3 = h5py.File(test_save_file, 'w')
f3.create_dataset('data', data=np.array(test_samples))

