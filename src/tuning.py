import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from sklearn.metrics import confusion_matrix

from models.CAE import CAE
from generate import arma
from train import Trainer

import hydra
from hydra.experimental import initialize, compose
import itertools

import optuna

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

def objective(trial):
    # HYPERPARAMETER SETTING
    # cfg_model.alpha = float(params[0])
    # cfg_model.lmd = trial.suggest_float("lmd", 1e-4, 1e-1, log=True)
    cfg_train.early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 10, 40)
    cfg_train.verbose=False
    # cfg_train.iters = 1
    # cfg_model.hidden_nn = params[1]
    # END

    model = CAE(cfg_model)

    trainer = Trainer(cfg_train)
    train_losses, valid_losses = trainer.fit(model, X_train, X_valid)

    inp, real = X_test[:,:,:-1], X_test[:,:,-1]
    pred, output = model(inp, get_training=True)
    pred = pred.detach().numpy()

    cors = [scipy.stats.spearmanr(pred[i,0], inp[i,0]).correlation for i in range(inp.shape[0])]
    cor = np.mean(cors)

    probs = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).detach().numpy()
    cm = confusion_matrix(real, pred)
    acc = np.sum(np.diag(cm))/np.sum(cm)

    return acc

X_train, X_valid, X_test = generate_data()

search_space = {"early_stopping_rounds": [10, 20, 30, 40]}
sampler = optuna.samplers.GridSampler(search_space)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_params)
optuna.visualization.plot_contour(study, params=["early_stopping_rounds"]).show()