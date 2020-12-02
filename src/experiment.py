import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, confusion_matrix

from models.CAE import CAE
from models.VCAE import VCAE
from generate import sin_cos, arma, wind
from train import train
from utils import latent_space, choose_bottleneck, compute_distance_matrix

import hydra
from hydra.experimental import initialize, compose
import itertools

torch.manual_seed(1)
np.random.seed(1)

with initialize(config_path="configs"):
    cfg = compose(config_name="config")

cfg_dataset, cfg_model, cfg_train = cfg.dataset, cfg.model, cfg.train

def generate_data():
    n_train, n_valid, n_test = cfg_dataset.n_train, cfg_dataset.n_valid, cfg_dataset.n_test
    length = cfg_model.length # each observation is a vector of size (1,length)

    n = n_train+n_valid+n_test

    X1 = arma(n//3, length, ar=[0, -0.5] , ma=[0, 0.1])
    X2 = arma(n//3, length, ar=[0, 0, 0.7] , ma=[0, 0, 0.05])
    X3 = arma(n//3, length, ar=[0, 0, 0, 0, -0.6] , ma=[0, 0, 0, 0, 0.2])
    class1 = np.array([0]*(n//3), dtype=np.float32).reshape(n//3, 1, 1)
    class2 = np.array([1]*(n//3), dtype=np.float32).reshape(n//3, 1, 1)
    class3 = np.array([2]*(n//3), dtype=np.float32).reshape(n//3, 1, 1)
    X1 = np.append(X1, class1, 2)
    X2 = np.append(X2, class2, 2)
    X3 = np.append(X3, class3, 2)

    X = np.r_[X1,X2,X3]

    idx = list(range(len(X)))
    np.random.shuffle(idx)
    X = X[idx]
    X = torch.from_numpy(X)

    X_train, X_valid, X_test = X[:n_train], X[n_train:n_train+n_valid], X[n_train+n_valid:]
    return X_train, X_valid, X_test

def execution(X_train, X_test, X_valid, params):
    # HYPERPARAMETER SETTING
    cfg_model.alpha = float(params[0])
    # cfg_model.hidden_nn = params[1]
    # END

    model = CAE(cfg_model)

    train_losses, valid_losses = train(model, cfg_train, X_train, X_valid)

    recon, pred = model(X_test, get_training=True)
    recon = recon.detach().numpy()

    cors = [scipy.stats.spearmanr(recon[i,0], X_test[i,0,:-1]).correlation for i in range(X_test.shape[0])]

    probs = torch.nn.functional.softmax(pred, dim=0)
    pred = torch.argmax(probs, dim=1).detach().numpy()
    real = X_test[:,:,-1].flatten().long().detach().numpy()
    cm = confusion_matrix(real, pred)
    acc = np.sum(np.diag(cm))/X_test.shape[0]

    print("Correlation, avg and std:", np.mean(cors), np.std(cors))
    print("Accuracy:", acc)

    return acc, np.mean(cors)

# HYPERPARAMETERS
alphas = np.linspace(0,1,20)
# alphas = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
neurons = list(range(10,40,5))

params = [alphas]
# alphas = np.linspace(0.4,0.6,5)
print(params)
# CONFIG
K = 5
cfg_train.early_stopping_rounds = 10
cfg_dataset.n_train, cfg_dataset.n_valid, cfg_dataset.n_test = 300, 100, 100
# cfg_train.iters = 1
# END

kaccs, kcors = [], []

for k in range(K): # number of repetitions
    X_train, X_valid, X_test = generate_data()
    accs, cors = [], []
    for param in itertools.product(*params):
        print(k, param)
        a, c = execution(X_train, X_valid, X_test, param)
        accs.append(a)
        cors.append(c)
    kaccs.append(accs)
    kcors.append(cors)

shape = tuple([K] + [len(p) for p in params])
kaccs = np.array(kaccs).reshape(shape)
kcors = np.array(kcors).reshape(shape)

accs_mean = np.mean(kaccs, axis=0)
accs_std = np.std(kaccs, axis=0)
cors_mean = np.mean(kcors, axis=0)
cors_std = np.std(kcors, axis=0)

print("RESULTS")
print()
print(params)
print()
print(kaccs)
print(kcors)

# ONE VARIABLE

if len(params) == 1:
    plt.plot(alphas, cors_mean, "o-", label="Correlation")
    plt.fill_between(alphas, cors_mean-cors_std, cors_mean+cors_std, alpha=0.1)
    plt.plot(alphas, accs_mean, "o-", label="Accuracy")
    plt.fill_between(alphas, accs_mean-accs_std, accs_mean+accs_std, alpha=0.1)
    plt.legend()
    plt.xlabel("alpha")
    plt.show()

# TWO VARIABLES

elif len(params) == 2:
    for i in range(len(alphas)):
        plt.plot(neurons, np.mean(kcors, axis=0)[i], "o-", label=f"alpha={alphas[i]}")
        plt.fill_between(neurons, cors_mean[i] - cors_std[i],
                                  cors_mean[i] + cors_std[i], alpha=0.1)
    plt.legend()
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("Correlation")
    plt.show()

    for i in range(len(alphas)):
        plt.plot(neurons, np.mean(kaccs, axis=0)[i], "s-", label=f"alpha={alphas[i]}")
        plt.fill_between(neurons, accs_mean[i] - accs_std[i],
                                  accs_mean[i] + accs_std[i], alpha=0.1)
    plt.legend()
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("Accuracy")
    plt.show()
