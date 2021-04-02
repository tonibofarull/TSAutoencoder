import torch
import numpy as np
from torch import optim


def get_hist(x):
    x = x.reshape(-1, 1)
    n = x.shape[0]
    counts, bins = np.histogram(x, bins="auto")  # TODO: check auto

    counts = n - counts
    probs = counts / sum(counts)
    return bins, probs


def sample_from_hist(hist, size=1):
    """
    Select bin with probability and uniformly selects and element of the bin
    """
    bins, probs = hist
    As = np.random.choice(a=range(len(probs)), p=probs, replace=True, size=size)
    elems = [np.random.uniform(bins[a], bins[a + 1]) for a in As]
    return np.array(elems)


# Shapley Values


def shapley_sampling(x, func, feature, histograms=None, n_batches=10, batch_size=64):
    """
    Importance of input with respect the output
    """
    length = x.shape[0]
    sv = torch.zeros(func(x).shape[-1])
    x = x.reshape(1, length).repeat(batch_size, 1)
    for _ in range(n_batches):
        # SAMPLING
        y = torch.zeros((batch_size, length))
        for i in range(length):
            y[:, i] = torch.tensor(sample_from_hist(histograms[i], size=batch_size))
        # END OF SAMPLING

        O = np.array([np.random.permutation(length) for _ in range(batch_size)])
        idx = np.where(O == feature)
        Os = [O[i, :j] for i, j in zip(idx[0], idx[1])]

        sel = torch.zeros((batch_size, length), dtype=torch.bool)
        sel[np.concatenate([np.repeat(i, len(Os[i])) for i in range(batch_size)]), np.concatenate(Os)] = True

        x2 = torch.where(sel, x, y)
        x1 = x2.clone()
        x1[:, feature] = x[:, feature]

        with torch.no_grad():
            v1 = func(x1)
            v2 = func(x2)
        sv += torch.sum(v1 - v2, axis=0)

    sv /= (n_batches * batch_size)
    return sv.numpy()


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
