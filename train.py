#!/usr/bin/python
#encoding=utf-8

from data.data_loader import *
from model import Model
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np

USE_CUDA=True

def train(train_dataloader, model, optimizer, loss_fn, print_every=50):
    model.train()
    
    total_loss = 0
    print_loss = 0
    i = 0
    for batch in train_dataloader:
        inputs, weights, labels = batch
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        
        #batch_size = len(outputs)
        loss = loss_fn(outputs, labels)
        #for x in range(batch_size):
        #    loss += weights[x]*loss_fn(outputs[x].view(1, -1), labels[x])
        #loss = loss / batch_size
        
        print_loss += loss.data[0]
        total_loss += loss.data[0]
        
        if (i+1) % print_every == 0:
            print("batch=%d, loss=%.4f" % ((i+1), print_loss / print_every))
            print_loss = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    average_loss = total_loss / i
    print("Epoch done, average_loss: %.4f" % average_loss)
    return average_loss

def dev(dev_dataloader, model, loss_fn):
    model.eval()
    
    total_loss = 0
    total_acc = 0
    i = 0
    for batch in dev_dataloader:
        inputs, weights, labels = batch
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        
        pred = torch.max(outputs, 1)[1]
        num_correct = (pred == labels).sum()
        batch_size = len(outputs)
        loss = loss_fn(outputs, labels)
        #for x in range(batch_size):
        #    loss += weights[x]*loss_fn(outputs[x].view(1, -1), labels[x])
        
        total_loss += loss.data[0]
        total_acc += float(num_correct.data[0]) / batch_size
        i += 1
    return total_loss / i, total_acc / i

def main():

    num_epoches = 100
    least_train_epoch = 30
    end_ajust_loss = 0.1
    loss_best = 100
    decay = 0.5
    batch_size = 256
    init_lr = 0.0001
    learning_rate = init_lr
    weight_decay = 0.0001
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    
    from visdom import Visdom
    viz = Visdom(env='fan')
    opts = [dict(title="stock training loss 20170916", ylabel='Loss', xlabel='Epoch'),
            dict(title="stock dev loss 20170916", ylabel='Loss', xlabel='Epoch'),
            dict(title="stock dev acc 20170916", ylabel='acc', xlabel='Epoch')]
    viz_window = [None, None, None]
    
    #加在数据集
    train_dataset = myDataset(date='20170916/', data_set="train", n_feats=88)
    train_dataloader = myDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=4, pin_memory=False)
    dev_dataset = myDataset(date='20170916/', data_set='dev', n_feats=88)
    dev_dataloader = myDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=4, pin_memory=False)
    
    #模型，目标函数，优化器定义 
    input_size = train_dataset.n_feats
    num_class = 2
    model = Model(net=[input_size, 16, 8, 4, num_class], dropout=0.4, bias=True)
    if USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay = weight_decay)
    loss_fn = nn.NLLLoss()

    #开始迭代
    start_time = time.time()
    count = 0
    train_loss = []
    dev_loss = []
    dev_acc = []
    while not stop_train:
        if count >= num_epoches:
            break
        count += 1
        if adjust_rate_flag:
            learning_rate *= decay
            for param in optimizer.param_groups:
                param['lr'] *= decay
        
        print("Start training epoch %d, learning_rate: %.5f" % (count, learning_rate))
        tloss = train(train_dataloader, model, optimizer, loss_fn)
        train_loss.append(tloss)

        dloss, dacc = dev(dev_dataloader, model, loss_fn)
        dev_loss.append(dloss)
        dev_acc.append(dacc)

        #if adjust_time == 3:
        #    stop_train = True

        #if count > least_train_epoch:
        if dloss < loss_best:
            best_model = model
            loss_best = dloss
        #    adjust_rate_flag = False
        #else:
        #    adjust_rate_flag = True
        #    model = best_model

        time_used = (time.time()-start_time) / 60
        print("Epoch %d done, cv loss is: %.4f, acc is: %.4f, time_used:%.4f minutes" % (count, dloss, dacc, time_used))
        x_axis = range(count)
        y_axis = [train_loss[0:count], dev_loss[0:count], dev_acc[0:count]]
        for x in range(len(viz_window)):
            if viz_window[x] is None:
                viz_window[x] = viz.line(X=np.array(x_axis), Y=np.array(y_axis[x]), opts = opts[x],)
            else:
                viz.line(X=np.array(x_axis), Y=np.array(y_axis[x]), win = viz_window[x], update='replace')

    print("End training.")
    best_path = './log/best_model_cv'+str(loss_best)+'.pkl'
    torch.save(best_model.save_package(best_model, optimizer, loss_results=train_loss, training_cer_results=dev_loss, dev_cer_results=dev_acc) , best_path)
if __name__ == '__main__':
    main()
