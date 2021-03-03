import numpy as np
import torch
import pandas as pd
import torch

"""
Output must be a numpy array
"""

def normalize(X):
    X = torch.clone(X)
    M, _ = torch.max(X[:,:,:-1], dim=2, keepdim=True)
    m, _ = torch.min(X[:,:,:-1], dim=2, keepdim=True)
    X[:,:,:-1] = (X[:,:,:-1]-m)/(M-m)
    return X


def ElectricDevices():
    df = pd.read_csv("../data/ElectricDevices/ElectricDevices_TRAIN.txt", sep="  ", header=None, engine="python")
    data_train = df.iloc[:,list(range(1,df.shape[1])) + [0]].to_numpy().reshape(-1, 1, df.shape[1])
    data_train = data_train.astype(np.float32)
    idx = list(range(len(data_train)))
    np.random.shuffle(idx)
    data_train = data_train[idx]
    data_train[:,0,-1] -= 1
    data_train, data_valid = data_train[:-1000,:,:], data_train[-1000:,:,:]

    df = pd.read_csv("../data/ElectricDevices/ElectricDevices_TEST.txt", sep="  ", header=None, engine="python")
    data_test = df.iloc[:,list(range(1,df.shape[1])) + [0]].to_numpy().reshape(-1, 1, df.shape[1])
    data_test = data_test.astype(np.float32)
    #idx = list(range(len(data_test)))
    #np.random.shuffle(idx)
    #data_test = data_test[idx]
    data_test[:,0,-1] -= 1
    
    data_train = torch.from_numpy(data_train)
    data_valid = torch.from_numpy(data_valid)
    data_test = torch.from_numpy(data_test)
    return data_train, data_valid, data_test
