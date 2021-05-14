import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from interpretability import shapley_sampling

# diverging_colors = sns.color_palette("RdBu", 9)
diverging_colors = sns.color_palette("vlag", as_cmap=True)


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


def shapley_input_vs_output(model, selected, X_test, hist_input):
    func = lambda x: model(x.reshape(-1, 1, 96), False)[0][:, 0]
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_input) for j in range(model.length)]
    )
    attrs = []
    for i, x in enumerate(selected):
        print(i)
        attrs.append(f_attrs(X_test[x, 0]))

    fig1, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(15, 15), constrained_layout=True
    )
    axs = np.array(axs)
    axs[2, 1].set_axis_off()
    axs[2, 2].set_axis_off()
    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Output Time Series", ylabel="Input Time Series")

        ax = axs.flatten()[i].twinx()
        sns.lineplot(
            x=range(96), y=X_test[x, 0], label="Original", color="green", ax=ax
        )
        sns.lineplot(
            x=range(96),
            y=model(X_test[x, 0].reshape(1, 1, -1), False)[0][0, 0].detach().numpy(),
            label="Predicted",
            color="purple",
            ax=ax,
        )
        ax.legend()
    plt.show()


def shapley_bottleneck_vs_output(model, selected, X_test, hist_bn):
    func = lambda x: model.decoder(x)[:, 0, :]
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_bn) for j in range(model.bottleneck_nn)]
    )
    attrs = []
    for i, x in enumerate(selected):
        print(i)
        inp = model.encoder(X_test[x, 0].reshape(1, 1, -1), False).flatten()
        attrs.append(f_attrs(inp))

    fig2, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(15, 15), constrained_layout=True
    )
    axs = np.array(axs)
    axs[2, 1].set_axis_off()
    axs[2, 2].set_axis_off()
    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Output Time Series", ylabel="Input Bottleneck")

        ax = axs.flatten()[i].twinx()
        sns.lineplot(
            x=range(96), y=X_test[x, 0], label="Original", color="green", ax=ax
        )
        sns.lineplot(
            x=range(96),
            y=model(X_test[x, 0].reshape(1, 1, -1), False)[0][0, 0].detach().numpy(),
            label="Predicted",
            color="purple",
            ax=ax,
        )
        ax.legend()
    plt.show()


def shapley_input_vs_bottleneck(model, selected, X_test, hist_input):
    func = lambda x: model.encoder(x.reshape(-1, 1, 96), False)
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_input) for j in range(model.length)]
    ).T
    attrs = []
    for i, x in enumerate(selected):
        print(i)
        inp = X_test[x, 0]
        attrs.append(f_attrs(inp))

    fig3, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(15, 15), constrained_layout=True
    )
    axs = np.array(axs)
    axs[2, 1].set_axis_off()
    axs[2, 2].set_axis_off()
    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Input Time Series", ylabel="Output Bottleneck")

        ax = axs.flatten()[i].twinx()
        sns.lineplot(
            x=range(96), y=X_test[x, 0], label="Original", color="green", ax=ax
        )
        sns.lineplot(
            x=range(96),
            y=model(X_test[x, 0].reshape(1, 1, -1), False)[0][0, 0].detach().numpy(),
            label="Predicted",
            color="purple",
            ax=ax,
        )
    plt.show()


def shapley_bottleneck_vs_class(model, selected, X_test, hist_bn):
    func = lambda x: model.classifier.get_probs(model.classifier(x.reshape(-1, 24)))
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_bn) for j in range(model.bottleneck_nn)]
    )
    attrs = []
    for i, x in enumerate(selected):
        print(i)
        inp = model.encoder(X_test[x, 0].reshape(1, 1, -1), False).flatten()
        attrs.append(f_attrs(inp))

    fig4, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), constrained_layout=True)
    axs = np.array(axs)
    axs[2, 1].set_axis_off()
    axs[2, 2].set_axis_off()
    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Output Class", ylabel="Input Bottleneck")
    plt.show()


def shapley_input_vs_class(model, selected, X_test, hist_input):
    func = lambda x: model.classifier.get_probs(
        model.classifier(model.encoder(x.reshape(-1, 1, 96), False))
    )
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_input) for j in range(model.length)]
    ).T
    attrs = []
    for i, x in enumerate(selected):
        print(i)
        inp = X_test[x, 0]
        attrs.append(f_attrs(inp))

    fig5, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(15, 15), constrained_layout=True
    )
    axs = np.array(axs)
    axs[2, 1].set_axis_off()
    axs[2, 2].set_axis_off()
    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Input Time Series", ylabel="Output Bottleneck")

        ax = axs.flatten()[i].twinx()
        sns.lineplot(
            x=range(96), y=X_test[x, 0], label="Original", color="green", ax=ax
        )
        sns.lineplot(
            x=range(96),
            y=model(X_test[x, 0].reshape(1, 1, -1), False)[0][0, 0].detach().numpy(),
            label="Predicted",
            color="purple",
            ax=ax,
        )
    plt.show()
