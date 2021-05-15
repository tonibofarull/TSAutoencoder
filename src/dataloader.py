import abc

import numpy as np
import pandas as pd
import torch


class Dataset(abc.ABC):
    def __init__(self):
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.load_data()

    @abc.abstractmethod
    def load_data(self):
        pass

    @staticmethod
    def normalize(X):
        X = torch.clone(X)
        M, _ = torch.max(X[:, :, :-1], dim=2, keepdim=True)
        m, _ = torch.min(X[:, :, :-1], dim=2, keepdim=True)
        X[:, :, :-1] = (X[:, :, :-1] - m) / (M - m)
        return X

    def __call__(self):
        return self.data_train, self.data_valid, self.data_test


class ElectricDevices(Dataset):
    def load_data(self):
        df = pd.read_csv(
            "../data/ElectricDevices/ElectricDevices_TRAIN.txt",
            sep="  ",
            header=None,
            engine="python",
        )
        data_train = (
            df.iloc[:, list(range(1, df.shape[1])) + [0]]
            .to_numpy()
            .reshape(-1, 1, df.shape[1])
        )
        data_train = data_train.astype(np.float32)
        idx = list(range(len(data_train)))
        np.random.shuffle(idx)
        data_train = data_train[idx]
        data_train[:, 0, -1] -= 1
        data_train, data_valid = data_train[:-1000, :, :], data_train[-1000:, :, :]

        df = pd.read_csv(
            "../data/ElectricDevices/ElectricDevices_TEST.txt",
            sep="  ",
            header=None,
            engine="python",
        )
        data_test = (
            df.iloc[:, list(range(1, df.shape[1])) + [0]]
            .to_numpy()
            .reshape(-1, 1, df.shape[1])
        )
        data_test = data_test.astype(np.float32)
        data_test[:, 0, -1] -= 1

        self.data_train = Dataset.normalize(torch.from_numpy(data_train))
        self.data_valid = Dataset.normalize(torch.from_numpy(data_valid))
        self.data_test = Dataset.normalize(torch.from_numpy(data_test))
