#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, cv2, csv
import numpy as np


# In[ ]:


def one_hot_encoding(text, allowedChars):
    label_list = []
    for c in text:
        onehot = [0] * len(allowedChars)
        onehot[allowedChars.index(c)] = 1
        label_list.append(onehot)
    return label_list


# In[ ]:


def one_hot_decoding(prediction, allowedChars):
    text = ''
    for predict in prediction:
        value = np.argmax(predict[0])
        text += allowedChars[value]
    return text


# In[ ]:


def read_train_data(filename, size):
    train_data = []
    if os.path.isdir(filename):
        train_data = np.stack([np.array(cv2.imread(filename + str(index) + ".jpg"))/255.0 for index in range(1, size + 1)])
    return train_data


# In[ ]:


def read_label_data(filename, allowedChars, num_dic):
    train_label = []
    traincsv = open(filename, 'r', encoding = 'utf8')
    
    read_label =  [one_hot_encoding(row[0], allowedChars) for row in csv.reader(traincsv)]
    train_label = [[] for _ in range(num_dic)]
    
    for arr in read_label:
        for index in range(num_dic):
            train_label[index].append(arr[index])
    train_label = [arr for arr in np.asarray(train_label)]
    return train_label


# In[ ]:


import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

