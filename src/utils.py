import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from models.CAE import CAE
from train import train


def latent_space(model, n):
    _, axs = plt.subplots(n, n, figsize=(15,15))

    for i, a in enumerate(np.linspace(0,1,n)):
        for j, b in enumerate(np.linspace(0,1,n)):
            bottleneck = torch.tensor([[a,b]], dtype=torch.float)
            out = model.decode(bottleneck)
            axs[i,j].axis("off")
            axs[i,j].set_title(f"({a:.2f},{b:.2f})")
            axs[i,j].plot(out[0,0].detach().numpy())
    plt.show()

def choose_bottleneck(X_test, X_train, X_valid, length, M, Lf):
    vals = []

    for bottleneck_nn in range(1,10):
        print("bottleneck:", bottleneck_nn)
        model = CAE(length=length, Lf=Lf, M=M, bottleneck_nn=bottleneck_nn)
        _, _ = train(model, X_train, X_valid, iters=3000, early_stopping_rounds=10, verbose=False)

        pred1 = model(X_test)

        pred1 = pred1.detach().numpy()

        cors = [scipy.stats.spearmanr(pred1[i,0], X_test[i,0]).correlation for i in range(X_test.shape[0])]
        vals.append(cors)
    return vals
