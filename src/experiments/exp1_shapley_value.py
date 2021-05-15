from IPython import get_ipython

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import sys

sys.path.append("..")

from itertools import chain, combinations
from math import factorial

import seaborn as sns
import torch
import numpy as np
from src.interpretability import sample_from_hist, get_hist
import random
import matplotlib.pyplot as plt

torch.manual_seed(4444)
np.random.seed(4444)
random.seed(4444)

sns.set(font_scale=1.75, style="white")


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def shapley_value(x, b, i, func, hist=None):
    n = len(x)
    l = list(range(n))
    del l[i]

    if hist is not None:
        b = torch.tensor(
            [sample_from_hist(hist[i]) for i in range(n)], dtype=torch.float32
        ).flatten()

    sv = 0
    for S in powerset(l):
        S = np.array(S).flatten()
        v1, v2 = b.clone(), b.clone()
        v1[i] = x[i]
        if len(S) != 0:
            v1[S] = x[S]
            v2[S] = x[S]
        const = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
        sv += const * (func(v1) - func(v2))
    return sv


# Hidden function


def func1():
    phi = torch.tensor([1, 2, 3, 4, 1])

    def foo(x):
        return torch.sum(phi * x)

    return foo


def func2():
    def foo(x):
        return torch.exp(-1 / 4 * torch.square(x[0])) + torch.exp(
            -1 / 4 * torch.square(x[1])
        )

    return foo


# Data generator


def generate1():
    locs = [1, 1, 1, 1, 15]
    x = np.concatenate(
        [np.random.normal(loc=i, scale=1, size=(1, 1)) for i in locs], axis=1
    )
    x = torch.tensor(x, dtype=torch.float32).flatten()
    return x


def generate2():
    a = torch.normal(mean=5, std=1, size=(1, 1))
    b = torch.normal(mean=-5, std=1, size=(1, 1))
    sel = torch.rand(1, 1) < 0.5
    x = torch.where(sel, a, b)
    x = torch.cat([x, torch.normal(mean=0, std=1, size=(1, 1))], dim=1)
    return x.flatten()


sns.set(font_scale=1.125, style="white")
x = np.abs(np.stack([generate2() for _ in range(100000)]))
sns.kdeplot(x[:, 0], label="First component")
sns.kdeplot(x[:, 1], label="Second component")
plt.legend()
plt.savefig("distributions.png", dpi=100)


X = np.arange(0, 10, 0.25)
Y = np.arange(0, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-1 / 4 * (X ** 2 + Y ** 2))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

from matplotlib import cm

ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.view_init(30, -60)
fig.savefig("a.png", dpi=100)

# Evaluate the shapley values


def compute(
    func=func1,
    gen=generate1,
    hist_input=True,
    alpha=1,
    b_zero=False,
    normalize=False,
    ax=None,
    title=None,
):
    n = 1000
    f = func()

    x = torch.stack([gen() for _ in range(n)])

    if normalize:
        x = x - torch.mean(x, dim=0) / torch.std(x, dim=0)

    if hist_input:
        hist_input = [get_hist(x[:, j], alpha=alpha) for j in range(x.shape[1])]
    else:
        hist_input = None

    # b = torch.tensor([0,0], dtype=torch.float32)
    # b = torch.tensor([0,0,0,0,0], dtype=torch.float32)
    if b_zero:
        b = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
    else:
        b = torch.mean(x, dim=0)

    res = [
        [
            shapley_value(x[j], b, i, f, hist=hist_input).item()
            for i in range(x.shape[1])
        ]
        for j in range(n)
    ]
    res = np.abs(np.array(res))

    data = {"x": res.flatten(), "class": np.tile(range(res.shape[1]), res.shape[0]) + 1}
    sns.boxplot(x="class", y="x", data=data, ax=ax)
    ax.set(xlabel="Feature", ylabel="Absolute Shapley Value", title=title)


# First experiment

fig5, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), constrained_layout=True)
compute(func=func1, gen=generate1, hist_input=False, b_zero=True, ax=axs)
fig5.savefig("base_zeros.png", dpi=100)


fig5, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), constrained_layout=True)
compute(
    func=func1,
    gen=generate1,
    hist_input=False,
    b_zero=False,
    ax=axs[0, 0],
    title="Mean baseline",
)
compute(
    func=func1,
    gen=generate1,
    hist_input=True,
    alpha=1,
    ax=axs[0, 1],
    title="Distribution baseline",
)
compute(
    func=func1,
    gen=generate1,
    hist_input=True,
    alpha=0,
    ax=axs[1, 0],
    title="Uniform baseline",
)
compute(
    func=func1,
    gen=generate1,
    hist_input=True,
    alpha=-1,
    ax=axs[1, 1],
    title="Inverse proportional baseline",
)
fig5.savefig("base.png", dpi=100)

# Second experiment

fig5, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10), constrained_layout=True)
compute(
    func=func2,
    gen=generate2,
    hist_input=False,
    b_zero=False,
    ax=axs[0, 0],
    title="Mean baseline",
)
compute(
    func=func2,
    gen=generate2,
    hist_input=True,
    alpha=1,
    ax=axs[0, 1],
    title="Distribution baseline",
)
compute(
    func=func2,
    gen=generate2,
    hist_input=True,
    alpha=0,
    ax=axs[1, 0],
    title="Uniform baseline",
)
compute(
    func=func2,
    gen=generate2,
    hist_input=True,
    alpha=-1,
    ax=axs[1, 1],
    title="Inverse proportional baseline",
)
fig5.savefig("base2.png", dpi=100)
