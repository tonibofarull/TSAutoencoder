import numpy as np
import torch
from torch import optim


def get_hist(x, alpha=1):
    x = x.reshape(-1, 1)
    counts, bins = np.histogram(x, bins="auto")  # TODO: check auto

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


# Shapley Values


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
    # print(np.std(phis, axis=0))
    phis = np.mean(phis, axis=0)
    return phis


# Feature Visualization


def feature_visualization(model, position, sel):
    """
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """
    X = torch.rand((1, 1, 96), requires_grad=True)
    optimizer = optim.LBFGS([X])

    def closure():
        optimizer.zero_grad()
        model.zero_grad()
        out = model(torch.sigmoid(X), apply_noise=False)[sel]
        y = -out[0, position]
        y.backward()
        return y

    for i in range(100):
        optimizer.step(closure)
    return torch.sigmoid(X).detach().numpy().flatten()
