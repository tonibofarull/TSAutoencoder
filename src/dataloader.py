import abc

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm

class Dataset(abc.ABC):
    """
    Base class for Dataset
    Custom class that a dataset has to inherit and implement its abstract methods
    """

    def __init__(self):
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.load_data()

    @abc.abstractmethod
    def load_data(self):
        """
        Load training, validation and testing and update:
        - self.data_train
        - self.data_valid
        - self.data_test

        Each variable has the shape (N, 1, length+1) where 'length' represents the length of the time series
        and the last column represents the class of the observation, starting from 0.
        """
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
        data_train = ElectricDevices.read_data(
            "../data/ElectricDevices/ElectricDevices_TRAIN.txt"
        )
        data_train = data_train.astype(np.float32)
        np.random.shuffle(data_train)
        data_train[:, 0, -1] -= 1

        data_train, data_valid = data_train[:-1000, :, :], data_train[-1000:, :, :]

        data_test = ElectricDevices.read_data(
            "../data/ElectricDevices/ElectricDevices_TEST.txt"
        )
        data_test = data_test.astype(np.float32)
        data_test[:, 0, -1] -= 1

        self.data_train = Dataset.normalize(torch.from_numpy(data_train))
        self.data_valid = Dataset.normalize(torch.from_numpy(data_valid))
        self.data_test = Dataset.normalize(torch.from_numpy(data_test))

    @staticmethod
    def read_data(file):
        """
        Reads data from 'file' and move the class index to the last column
        """
        df = pd.read_csv(file, sep="  ", header=None, engine="python")
        data = (
            df.iloc[:, list(range(1, df.shape[1])) + [0]]
            .to_numpy()
            .reshape(-1, 1, df.shape[1])
        )
        return data

class ARMA(Dataset):
    def __init__(self, case=4):
        self.case = case
        super().__init__()

    def load_data(self, n=900, L=96):
        data = None
        if self.case == 1:
            data = np.c_[ARMA.arma(n, L, [0, 0.75], [0, 0.35]), np.zeros(n)]
        elif self.case == 2:
            data = np.c_[ARMA.arma(n, L, [0, 0, 0.75], [0, 0, 0.35]), np.zeros(n)]
        elif self.case == 3:
            data = np.c_[ARMA.arma(n, L, [0, 0, 0, 0, 0.75], [0, 0, 0, 0, 0.35]), np.zeros(n)]
        elif self.case == 4:
            data = np.c_[ARMA.arma(n, L, [0, 0, 0, 0, 0, 0, 0.75], [0, 0, 0, 0, 0, 0, 0.35]), np.zeros(n)]
        else:
            data = np.r_[
                np.c_[ARMA.arma(n, L, [0, 0.75], [0, 0.35]), np.zeros(n)], 
                np.c_[ARMA.arma(n, L, [0, 0, 0.75], [0, 0, 0.35]), np.ones(n)],
                np.c_[ARMA.arma(n, L, [0, 0, 0, 0, 0.75], [0, 0, 0, 0, 0.35]), np.ones(n)*2],
                np.c_[ARMA.arma(n, L, [0, 0, 0, 0, 0, 0, 0.75], [0, 0, 0, 0, 0, 0, 0.35]), np.ones(n)*3]
            ]

        np.random.shuffle(data)
        data = data.reshape(-1, 1, L+1).astype(np.float32)
        data_train, data_valid, data_test = np.split(data, 3)

        self.data_train = Dataset.normalize(torch.from_numpy(data_train))
        self.data_valid = Dataset.normalize(torch.from_numpy(data_valid))
        self.data_test = Dataset.normalize(torch.from_numpy(data_test))

    @staticmethod
    def arma(n, length, ar, ma):
        ar = np.r_[1, -np.array(ar)]
        ma = np.r_[1,  np.array(ma)]
        arma_process = sm.tsa.ArmaProcess(ar, ma)
        data = [arma_process.generate_sample(nsample=length) for _ in range(n)]
        return np.array(data, dtype=np.float32)
