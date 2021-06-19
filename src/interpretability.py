import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from utils import get_subplots

# diverging_colors = sns.color_palette("RdBu", 9)
diverging_colors = sns.color_palette("vlag", as_cmap=True)


def get_hist(x, alpha=1):
    x = x.reshape(-1, 1)
    counts, bins = np.histogram(x, bins="auto")

    if alpha < 0:
        counts += 1
    probs = counts / sum(counts)
    probs = probs ** alpha / sum(probs ** alpha)
    return bins, probs


def sample_from_hist(hist, size=1):
    """
    Select bin with probability and uniformly selects and element of the bin
    """
    bins, probs = hist
    As = np.random.choice(a=range(len(probs)), p=probs, replace=True, size=size)
    pos = np.random.uniform(low=0, high=1, size=size)
    lows, highs = bins[As], bins[As + 1]
    return np.array((1 - pos) * lows + pos * highs)


"""
Global interpretability
"""


def global_interpretability(model):
    num_filter = model.k * model.M
    w_per_filter = model.length
    num_neurons = model.bottleneck_nn
    M = model.M

    weights = model.encoder.fc_conv_bn.weight
    heatmap = [
        [
            torch.mean(
                torch.abs(weights[j, i * w_per_filter : (i + 1) * w_per_filter])
            ).item()
            for i in range(num_filter)
        ]
        for j in range(num_neurons)
    ]
    heatmap = np.array(heatmap)
    x_axis_labels = [f"{i}-d:{model.dilation[i//M]}" for i in range(heatmap.shape[1])]

    _ = sns.heatmap(heatmap, xticklabels=x_axis_labels, cmap="gray", vmin=0)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig("global_interpretability.png", dpi=100)
    plt.show()


"""
Local interpretability
"""


def shapley_sampling(
    x, func, feature, histograms=None, n_y=10, n_batches=5, batch_size=64
):
    """
    Importance of input with respect the output
    """
    length = x.shape[-1]
    shape_out = func(x).shape[-1]
    x = x.float().reshape(1, length).repeat(batch_size, 1)

    if histograms is None:
        ys = torch.zeros((n_y, length))
    else:
        ys = torch.tensor(
            [sample_from_hist(histograms[i], size=n_y) for i in range(length)],
            dtype=torch.float32,
        ).T

    phis = []

    for y in ys:
        y = y.reshape(1, -1)
        sv = torch.zeros(shape_out)
        for _ in range(n_batches):
            # List of permutations
            perms = np.array([np.random.permutation(length) for _ in range(batch_size)])
            # List of, for every row, where the index of the feature is
            idx = np.where(perms == feature)  # ([0,1,2,...], [3,7,2,...])
            # List containing different sets S
            perms_S = [perms[i, :j] for i, j in zip(idx[0], idx[1])]

            # Vector indicating if select from X or from B (baseline)
            sel = torch.zeros((batch_size, length), dtype=torch.bool)  # Matrix of False
            rows = np.concatenate(
                [np.repeat(i, len(perms_S[i])) for i in range(batch_size)]
            )
            cols = np.concatenate(perms_S)
            sel[rows, cols] = True

            # Select positions of X where index is before in the permutation
            x2 = torch.where(sel, x, y)
            x1 = x2.clone()
            # For X1, also include the feature position
            x1[:, feature] = x[:, feature]

            with torch.no_grad():
                v1 = func(x1)
                v2 = func(x2)
            sv += torch.sum(v1 - v2, axis=0)
        sv /= n_batches * batch_size
        phis.append(sv.numpy())

    phis = np.stack(phis)
    phis = np.mean(phis, axis=0)
    return phis


# Shapley Values for different combinations of input, bottleneck, ouput and class prediction


def shapley_input_vs_output(
    model, selected, X_test, hist_input, nrows=3, ncols=3
):
    func = lambda x: model(x.reshape(-1, 1, 96), False)[0][:, 0]
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_input) for j in range(model.length)]
    )
    attrs = []
    for i, x in enumerate(selected):
        print(i, end=" ")
        attrs.append(f_attrs(X_test[x, 0]))

    fig, axs = get_subplots(nrows, ncols, (5*ncols, 5*nrows), selected)

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
    fig.savefig("input-output_dist.png", dpi=100)
    plt.show()


def shapley_bottleneck_vs_output(
    model, selected, X_test, hist_bn, nrows=3, ncols=3
):
    func = lambda x: model.decoder(x)[:, 0, :]
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_bn) for j in range(model.bottleneck_nn)]
    )
    attrs = []
    for i, x in enumerate(selected):
        print(i, end=" ")
        inp = model.encoder(X_test[x, 0].reshape(1, 1, -1), False).flatten()
        attrs.append(f_attrs(inp))

    fig, axs = get_subplots(nrows, ncols, (5*ncols, 5*nrows), selected)

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
    fig.savefig("bottleneck-output_dist.png", dpi=100)
    plt.show()


def shapley_input_vs_bottleneck(
    model, selected, X_test, hist_input, nrows=3, ncols=3
):
    func = lambda x: model.encoder(x.reshape(-1, 1, 96), False)
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_input) for j in range(model.length)]
    ).T
    attrs = []
    for i, x in enumerate(selected):
        print(i, end=" ")
        inp = X_test[x, 0]
        attrs.append(f_attrs(inp))

    fig, axs = get_subplots(nrows, ncols, (5*ncols, 5*nrows), selected)

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
    fig.savefig("input-bottleneck_dist.png", dpi=100)
    plt.show()


def shapley_bottleneck_vs_class(
    model, selected, X_test, hist_bn, nrows=3, ncols=3
):
    func = lambda x: model.classifier.get_probs(model.classifier(x.reshape(-1, 24)))
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_bn) for j in range(model.bottleneck_nn)]
    )
    attrs = []
    for i, x in enumerate(selected):
        print(i, end=" ")
        inp = model.encoder(X_test[x, 0].reshape(1, 1, -1), False).flatten()
        attrs.append(f_attrs(inp))

    fig, axs = get_subplots(nrows, ncols, (3*ncols, 3*nrows), selected)

    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Output Class", ylabel="Input Bottleneck")
    fig.savefig("bottleneck-class_dist.png", dpi=100)
    plt.show()


def shapley_input_vs_class(
    model, selected, X_test, hist_input, nrows=3, ncols=3
):
    func = lambda x: model.classifier.get_probs(
        model.classifier(model.encoder(x.reshape(-1, 1, 96), False))
    )
    f_attrs = lambda inp: np.array(
        [shapley_sampling(inp, func, j, hist_input) for j in range(model.length)]
    ).T
    attrs = []
    for i, x in enumerate(selected):
        print(i, end=" ")
        inp = X_test[x, 0]
        attrs.append(f_attrs(inp))

    fig, axs = get_subplots(nrows, ncols, (5*ncols, 5*nrows), selected)

    for i, x in enumerate(selected):
        ax = sns.heatmap(
            attrs[i],
            ax=axs.flatten()[i],
            center=0,
            cmap=diverging_colors,
            cbar_kws={"orientation": "horizontal"},
        )
        ax.set(xlabel="Input Time Series", ylabel="Output Class")

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
    fig.savefig("input-class_dist.png", dpi=100)
    plt.show()
