#!/usr/bin/python
#encoding=utf-8

import csv
import random
import h5py
import numpy as np

train_file = './20170916/ai_challenger_stock_train_20170916/stock_train_data_20170916.csv'
test_file = './20170916/ai_challenger_stock_test_20170916/stock_test_data_20170916.csv'
train_save_file = './20170916/min_max/train.h5py'
dev_save_file = './20170916/min_max/dev.h5py'
test_save_file = './20170916/min_max/test.h5py'

all_feature = []
#1-20随机抽取4个时间分成验证集
choose = random.sample(range(1,21),4)

#处理训练集
f = open(train_file, 'r')
reader = csv.reader(f)
training_samples = []
i = 0
max_group = 0
pos = 0
neg = 0
for row in reader:
    if i == 0:
        i += 1
        continue
    sample = []
    for idx in range(1, len(row)):
        sample.append(float(row[idx]))
    if int(row[-2]) > max_group:
        max_group = int(row[-2])
    training_samples.append(sample[88:])
    all_feature.append(sample[:88])
    i += 1
    print("process:",i)
f.close()

#处理测试集
f = open(test_file, 'r')
reader = csv.reader(f)
i = 0
test_samples = []
for row in reader:
    if i == 0:
        i += 1
        continue
    sample = []
    for idx in range(len(row)):
        sample.append(float(row[idx]))
    if int(row[-1]) > max_group:
        max_group = int(row[-1])
    test_samples.append([sample[0],sample[-1]])
    all_feature.append(sample[1:-1])
    i += 1
    print('process:',i)
f.close()

#数据归一化，减均值，处方差
all_feature = np.array(all_feature).astype(float)
x = np.rollaxis(all_feature, axis=0)
pmax = np.max(all_feature, axis=0)
pmin = np.min(all_feature, axis=0)
x = (x-pmin)/(pmax-pmin)
all_feature = x
#x = np.rollaxis(all_feature, axis=0)
#x -= np.mean(all_feature, axis=0)
#x /= np.std(all_feature, axis=0)

#分出验证集进行存储
train_pos = 0
train_neg = 0
n_training = len(training_samples)
training_samples = np.concatenate((all_feature[:n_training], np.array(training_samples)), axis=1)
test_samples = np.concatenate((all_feature[n_training:], np.array(test_samples)), axis=1)
train_samples = []
dev_samples = []
for x in range(n_training):
    if int(training_samples[x][-1]) == 20:
        dev_samples.append(training_samples[x][:-1])
        if int(training_samples[x][-3]) == 1:
            train_pos += 1
        else:
            train_neg += 1
    else:
        train_samples.append(training_samples[x][:-1])
        if int(training_samples[x][-3]) == 1:
            pos += 1
        else:
            neg += 1
dev_samples = np.array(dev_samples)
train_samples = np.array(train_samples)
print(train_samples.shape)
print(dev_samples.shape)
f1 = h5py.File(train_save_file, 'w')
f1.create_dataset('data', data=train_samples)
f2 = h5py.File(dev_save_file, 'w')
f2.create_dataset('data', data=np.array(dev_samples))
f3 = h5py.File(test_save_file, 'w')
f3.create_dataset('data', data=np.array(test_samples))
print("max_group:", max_group)
print(pos, neg)
print(train_pos, train_neg)

