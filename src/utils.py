import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pingouin import distance_corr  # Szekely and Rizzo
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def get_predictions(model, X_test):
    X_testp, outclass_testp, bn = model(X_test, apply_noise=False)
    X_testp = X_testp.detach().numpy()
    probs_testp = model.classifier.get_probs(outclass_testp)
    y_testp = torch.argmax(probs_testp, dim=1).detach().numpy()
    return X_testp, y_testp


def reconstruction(X_test, X_testp):
    cors = [
        distance_corr(X_testp[i, 0], X_test[i, 0].detach().numpy(), n_boot=None)
        for i in range(X_test.shape[0])
    ]
    print("Distance Correlation avg and std:", np.mean(cors), np.std(cors))
    print(
        "NRMSE:",
        (
            torch.sqrt(torch.mean(torch.square(X_test - X_testp)))
            / (torch.max(X_test) - torch.min(X_test))
        ).item(),
    )
    print()


def accuracy(y_test, y_testp):
    cm = confusion_matrix(y_test, y_testp)
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    print("Accuracy:", np.sum(np.diag(cm)) / np.sum(cm))
    plt.savefig("accuracy.png", dpi=100)
    plt.show()


def observation_reconstruction(selected, X_test, X_testp, y_test, y_testp, ncols=3):
    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(selected)/ncols)), ncols=ncols, figsize=(15, 11), constrained_layout=True
    )
    axs = np.array(axs).reshape(-1, ncols)
    for i, x in enumerate(selected):
        sns.lineplot(
            x=range(96), y=X_test[x, 0], label="Original", color="green", ax=axs.flatten()[i]
        )
        sns.lineplot(
            x=range(96),
            y=X_testp[x, 0],
            label="Predicted",
            color="purple",
            ax=axs.flatten()[i],
        )
    plt.savefig("reconstruction.png", dpi=100)
    plt.show()


# Data exploration


def data_input_exploration(X_train):
    sns.set(font_scale=1.125, style="white")

    ax = sns.histplot(X_train.flatten(), stat="density")
    ax.set(xlabel="Training input values")
    plt.plot()
    # plt.savefig("sv_data-distribution.png", dpi=100)


def data_bottleneck_exploration(model, X_train):
    _, _, bn = model(X_train, False)
    bn = bn.detach().numpy()

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(25, 20), constrained_layout=True)
    axs[4, 4].set_axis_off()
    for i in range(24):
        aux = pd.DataFrame({"x": bn[:, i]})
        axs.flat[i].set_title(f"Neuron {i}")
        sns.histplot(data=aux, x="x", ax=axs.flat[i], kde=True)
    plt.show()
    # plt.savefig("sv_data-distribution.png", dpi=100)


def baseline(data_train, data_valid, data_test):
    X_train, y_train = data_train[:, 0, :-1], data_train[:, 0, -1].numpy()
    X_valid, y_valid = data_valid[:, 0, :-1], data_valid[:, 0, -1].numpy()
    X_test, y_test = data_test[:, 0, :-1], data_test[:, 0, -1].numpy()
    y_train.astype(int)
    y_valid.astype(int)
    y_test.astype(int)

    best_acc = 0
    best_k = 0
    for i in range(1, 50):

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, y_train)

        y_validp = neigh.predict(X_valid)
        y_validp.astype(int)

        acc = np.sum(y_validp == y_valid) / len(y_valid)
        if best_acc < acc:
            best_acc = acc
            best_k = i
    print("Best k:", best_k)
    print("Best acc:", best_acc)

    neigh = KNeighborsClassifier(n_neighbors=best_k)
    X_aux = np.r_[X_train, X_valid]
    y_aux = np.r_[y_train, y_valid]
    neigh.fit(X_aux, y_aux)

    y_testp = neigh.predict(X_test)
    y_testp.astype(int)

    cm = confusion_matrix(y_test, y_testp)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    print("Test acc:", np.sum(np.diag(cm)) / np.sum(cm))
