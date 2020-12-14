import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from sklearn.metrics import confusion_matrix

from models.CAE import CAE
from models.VCAE import VCAE
from generate import sin_cos, arma, wind
from train import train

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
    # cfg_model.alpha = float(params[0])
    cfg_model.reg = float(params[0])
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
    acc = np.sum(np.diag(cm))/np.sum(cm)

    print("Correlation, avg and std:", np.mean(cors), np.std(cors))
    print("Accuracy:", acc)


    num_filter = len(model.dilation)*cfg_model.M
    w_per_filter = cfg_model.length-cfg_model.Lf+1 # weights per filter
    num_neurons = cfg_model.bottleneck_nn
    M = cfg_model.M 

    w = np.array([[torch.mean(torch.abs(model.full1.weight[j,i*w_per_filter:(i+1)*w_per_filter])).item() for i in range(num_filter)] for j in range(num_neurons)])

    x_axis_labels = [f"{i}-d:{model.dilation[i//M]}" for i in range(w.shape[1])] # number of filter - d:dilatation
    sns.heatmap(w, cmap="coolwarm", xticklabels=x_axis_labels) # y-axis => neuron of the bottleneck, x-axis => each position is one filter ordered by dilatation
    plt.savefig(f"{cfg_model.reg}.png")
    plt.close()

    return acc, np.mean(cors)

# HYPERPARAMETERS
# alphas = np.linspace(0,1,20)
# alphas = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
# neurons = list(range(10,40,5))
reg = [0.001, 0.002, 0.003, 0.004]

params = [reg]
# alphas = np.linspace(0.4,0.6,5)
print(params)
# CONFIG
K = 1
# cfg_train.early_stopping_rounds = 10
# cfg_dataset.n_train, cfg_dataset.n_valid, cfg_dataset.n_test = 300, 100, 100
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
    par1 = params[0]

    plt.plot(params[0], cors_mean, "o-", label="Correlation")
    plt.fill_between(params[0], cors_mean-cors_std, cors_mean+cors_std, alpha=0.1)
    plt.plot(params[0], accs_mean, "o-", label="Accuracy")
    plt.fill_between(params[0], accs_mean-accs_std, accs_mean+accs_std, alpha=0.1)
    plt.legend()
    plt.xlabel("alpha")
    plt.show()

# TWO VARIABLES

elif len(params) == 2:
    par1 = params[0]
    par2 = params[1]

    for i in range(len(par1)):
        plt.plot(par2, np.mean(kcors, axis=0)[i], "o-", label=f"alpha={par1[i]}")
        plt.fill_between(par2, cors_mean[i] - cors_std[i],
                               cors_mean[i] + cors_std[i], alpha=0.1)
    plt.legend()
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("Correlation")
    plt.show()

    for i in range(len(par1)):
        plt.plot(par2, np.mean(kaccs, axis=0)[i], "s-", label=f"alpha={par1[i]}")
        plt.fill_between(par2, accs_mean[i] - accs_std[i],
                               accs_mean[i] + accs_std[i], alpha=0.1)
    plt.legend()
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("Accuracy")
    plt.show()