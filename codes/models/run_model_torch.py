import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from itertools import product

import nn_torch

def get_NN(input_size):
    net = nn_torch.NN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    return net, criterion, optimizer

def get_loader(X_train, y_train, X_test, y_test):
    train = data_utils.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).view(len(y_train), -1))
    test = data_utils.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).view(len(y_test), -1))

    train_loader = data_utils.DataLoader(train, batch_size=12, shuffle=True)
    test_loader = data_utils.DataLoader(test, batch_size=12, shuffle=False)
    return train_loader, test_loader

def training(X_train, y_train, X_test, y_test):
    net, criterion, optimizer = get_NN(X_train.shape[1])
    losses = []
    train_accuracy = []
    test_accuracy = []
    for epoch in range(10):
        running_loss = 0.0
        train_loader, _ = get_loader(X_train, y_train, X_test, y_test)
        for i, (samples, labels) in enumerate(train_loader):
            net.zero_grad()
            input = torch.Tensor(samples)
            target = torch.Tensor(labels)
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        y_pred_train = np.round(net(torch.Tensor(X_train)).data.numpy())
        y_pred_test = np.round(net(torch.Tensor(X_test)).data.numpy())
        #print('epoch %d : loss=%.4f, acc=%.4f, val_acc=%.4f' % (epoch+1, 
        #                                            running_loss / len(train_loader),
        #                                            accuracy_score(y_train, y_pred_train), 
        #                                            accuracy_score(y_test, y_pred_test)))
        losses += [running_loss / len(train_loader)]
        train_accuracy += [accuracy_score(y_train, y_pred_train)]
        test_accuracy += [accuracy_score(y_test, y_pred_test)]
    return losses, train_accuracy, test_accuracy