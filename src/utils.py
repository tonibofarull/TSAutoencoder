import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns


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
