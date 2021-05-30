"""
First factorial plane representation using PCA and CAE
"""

import os
import random
import sys

sys.path.append("..")

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataloader import ARMA
from hydra.experimental import compose
from hydra.experimental import initialize_config_dir
from models.CAE import CAE
from train import Trainer
from utils import get_predictions
from sklearn.decomposition import PCA
import seaborn as sns

torch.manual_seed(4444)
np.random.seed(4444)
random.seed(4444)

plt.rcParams.update({"font.size": 12})


def with_cae(cfg, data_train, data_valid, X_test, y_test):
    model = CAE(cfg.model)

    trainer = Trainer(cfg.train)
    trainer.fit(model, data_train, data_valid)

    _, _, bn = get_predictions(model, X_test)

    bns = bn.detach().numpy()
    xs, ys = bns[:, 0], bns[:, 1]
    c = y_test.detach().numpy().flatten()
    sns.scatterplot(x=xs, y=ys, hue=c.astype(int))
    plt.xlabel("Neuron 1")
    plt.ylabel("Neuron 2")
    plt.savefig("bottleneck.png", dpi=100)
    plt.close()


def with_pca(X_train, X_test, y_test):
    pca = PCA()
    pca.fit(X_train.reshape(-1, 96))

    rep = pca.transform(X_test.reshape(-1, 96))
    xs, ys = rep[:, 0], rep[:, 1]
    c = y_test.detach().numpy().flatten()
    sns.scatterplot(x=xs, y=ys, hue=c.astype(int))
    plt.xlabel(f"Dim 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"Dim 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.savefig("factorialplane.png", dpi=100)
    plt.close()

    # Screeplot
    print(np.cumsum(pca.explained_variance_ratio_[:16]))
    plt.plot(pca.explained_variance_ratio_[:16], "o-")
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    plt.savefig("screeplot.png", dpi=100)
    plt.close()


def main():
    with initialize_config_dir(config_dir=os.path.abspath("../configs")):
        cfg = compose(config_name="arma5")

    cfg.model.bottleneck_nn = 2

    dl = ARMA(5)

    data_train, data_valid, data_test = dl()
    X_train, _ = data_train[:, :, :-1], data_train[:, :, -1]
    X_test, y_test = data_test[:, :, :-1], data_test[:, :, -1]

    with_cae(cfg, data_train, data_valid, X_test, y_test)
    with_pca(X_train, X_test, y_test)


if __name__ == "__main__":
    main()
