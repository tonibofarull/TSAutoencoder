import torch
import numpy as np

from models.CAE import CAE
from train import Trainer

import hydra
from hydra.experimental import initialize, compose
from dataloader import ElectricDevices, normalize

from sklearn.metrics import confusion_matrix
import dcor

import optuna

torch.manual_seed(1)
np.random.seed(1)
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

def objective(trial, data_train, data_valid, cfg):
    cfg_dataset, cfg_model, cfg_train = cfg.dataset, cfg.model, cfg.train
    
    # HYPERPARAMETER SETTING
    cfg_model.lmd = trial.suggest_loguniform("lmd", 1e-8, 1e-3)
    cfg_train.early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 5, 30)
    cfg_train.lr = trial.suggest_loguniform("lr", 1e-6, 1e-1)
    # END

    n_valid = data_valid.shape[0]
    data_valid1 = data_valid[:n_valid//2]
    data_valid2 = data_valid[n_valid//2:]

    model = CAE(cfg_model, num_classes=7)
    trainer = Trainer(cfg_train)
    trainer.fit(model, data_train, data_valid1)

    loss = model.loss(data_valid2, do_reg=False)

    return loss

def tuning(alpha, n_trial, study_name):

    with initialize(config_path="configs"):
        cfg = compose(config_name="config")

    cfg.model.alpha = alpha

    data_train_ori, data_valid_ori, _ = ElectricDevices()
    data_train, data_valid = normalize(data_train_ori), normalize(data_valid_ori)

    study = optuna.load_study(study_name=study_name, storage="sqlite:///storage.db")
    study.optimize(lambda trial : objective(trial, data_train, data_valid, cfg), n_trials=n_trial)

    return study

def acc_cor(alpha, hp):

    with initialize(config_path="configs"):
        cfg = compose(config_name="config")

    cfg_dataset, cfg_model, cfg_train = cfg.dataset, cfg.model, cfg.train

    cfg_model.alpha = alpha
    cfg_model.lmd = hp["lmd"]
    cfg_train.early_stopping_rounds = hp["early_stopping_rounds"]
    cfg_train.lr = hp["lr"]
    print("Acc and Cor with: alpha =", alpha, hp)

    data_train_ori, data_valid_ori, data_test_ori = ElectricDevices()
    data_train, data_valid, data_test = normalize(data_train_ori), normalize(data_valid_ori), normalize(data_test_ori)
    X_test, y_test = data_test[:,:,:-1], data_test[:,:,-1]

    model = CAE(cfg_model, num_classes=7)
    trainer = Trainer(cfg_train)
    trainer.fit(model, data_train, data_valid)

    X_test, y_test = data_test[:,:,:-1], data_test[:,:,-1].numpy()
    X_testp, outclass_testp, bn = model(X_test)
    X_testp = X_testp.detach().numpy()
    probs_testp = torch.nn.functional.softmax(outclass_testp, dim=1)
    y_testp = torch.argmax(probs_testp, dim=1).detach().numpy()

    cor = np.mean(dcor.rowwise(dcor.distance_correlation, X_testp[:,0], X_test[:,0].detach().numpy()))

    cm = confusion_matrix(y_test, y_testp)
    acc = np.sum(np.diag(cm))/np.sum(cm)

    return acc, cor

def main(
    alphas=[float(f) for f in np.linspace(0, 1, 20)],
    n_trial_per_job=4,
    n_jobs=8
):
    hyper_param = []
    for exp, alpha in enumerate(alphas):
        study_name = f"exp-{exp}"

        try:
            optuna.delete_study(study_name=study_name, storage="sqlite:///storage.db")
        except:
            pass
        optuna.create_study(study_name=study_name, storage="sqlite:///storage.db")
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(delayed(tuning)(alpha, n_trial_per_job, study_name) for _ in range(n_jobs))
        res = optuna.load_study(study_name=study_name, storage="sqlite:///storage.db")

        print(f"alpha: {alpha}, Best value: {res.best_value}, Best params: {res.best_params}")
        print()
        print()

        hyper_param.append(res)

    accs, cors = [], []

    for hp, alpha in zip(hyper_param, alphas):
        acc, cor = acc_cor(alpha, hp.best_params)
        accs.append(acc)
        cors.append(cor)

    plt.plot(alphas, cors, "o-", label="Correlation")
    plt.plot(alphas, accs, "o-", label="Accuracy")
    plt.legend()
    plt.xlabel("alpha")
    plt.savefig("alpha-cor_acc.png")

if __name__ == "__main__":
    main()
