import argparse
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dataloader import ElectricDevices
from hydra.experimental import compose
from hydra.experimental import initialize_config_dir
from interpretability import *
from models.CAE import CAE
from pingouin import distance_corr  # Szekely and Rizzo
from sklearn.metrics import confusion_matrix
from train import Trainer
from utils import baseline

torch.manual_seed(4444)
np.random.seed(4444)
random.seed(4444)

from utils import shapley_input_vs_output
from utils import shapley_bottleneck_vs_output
from utils import shapley_input_vs_bottleneck
from utils import shapley_bottleneck_vs_class
from utils import shapley_input_vs_class
from utils import shapley_bottleneck_vs_output

SELECTED = [3279, 1156, 7419, 5046, 3323, 6485, 5497]

# Autoencoder


def predict(model, data_test):
    X_test, y_test = data_test[:, :, :-1], data_test[:, :, -1].numpy()
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
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    print("Accuracy:", np.sum(np.diag(cm)) / np.sum(cm))
    plt.show()


def observation_reconstruction(selected, X_test, X_testp, y_test, y_testp):
    fig, axs = plt.subplots(
        nrows=2, ncols=len(selected), figsize=(25, 5), constrained_layout=True
    )
    for i, x in enumerate(selected):
        axs[0, i].set_title(f"Real class: {int(y_test[x][0])}")
        axs[0, i].plot(X_test[x, 0])
        axs[0, i].axis("off")
        axs[0, i].set_ylim((0, 1))

        axs[1, i].set_title(f"Predicted class: {int(y_testp[x])}")
        axs[1, i].plot(X_testp[x, 0])
        axs[1, i].axis("off")
        axs[1, i].set_ylim((0, 1))

        print("cor:", distance_corr(X_testp[x, 0], X_test[x, 0], n_boot=None))
    plt.show()


# Feature Visualization


def input_max_neuron(model):
    for i in range(24):
        a1 = feature_visualization(model, i, 2)
        plt.plot(a1, "o-")
        plt.title(f"Neuron {i}")
        plt.ylim(-0.05, 1.05)
        plt.savefig(f"../plots/{i}.png")
        plt.close()


def neuron_max_class(model):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(25, 10), constrained_layout=True)
    for i in range(7):  # selected):
        a1 = feature_visualization(model, i, 1)
        axs.flat[i].plot(a1)
        axs.flat[i].set_title(f"Representant class {i}")
        axs.flat[i].set_ylim(-0.05, 1.05)
    plt.show()


# Interpretability


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
    plt.show()


# Main


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fn",
        help="Filepath to .onnx, .tflite or SavedModel.",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    with initialize_config_dir(config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="config")

    dl = ElectricDevices()
    data_train, data_valid, data_test = dl()

    X_train, y_train = data_train[:, :, :-1], data_train[:, :, -1]
    X_valid, y_valid = data_valid[:, :, :-1], data_valid[:, :, -1]
    X_test, y_test = data_test[:, :, :-1], data_test[:, :, -1]

    model = CAE(cfg.model, num_classes=cfg.model.num_classes)
    # torch.save(model.state_dict(), "../weights/mod.pth")
    model.load_state_dict(torch.load("../weights/mod.pth"))

    X_testp, y_testp = predict(model, data_test)

    reconstruction(X_test, X_testp)
    accuracy(y_test, y_testp)

    # SELECTED = [
    #     np.random.choice([i for i, x in enumerate(y_test) if int(x) == j])
    #     for j in range(cfg.model.num_classes)
    # ]
    observation_reconstruction(SELECTED, X_test, X_testp, y_test, y_testp)

    # INTERPRETABILITY
    global_interpretability(model)

    hist_input = [get_hist(X_train[:, 0, i]) for i in range(cfg.model.length)]
    aux = model.encoder(X_train, False).detach().numpy()
    hist_bn = [get_hist(aux[:, i]) for i in range(cfg.model.bottleneck_nn)]

    shapley_input_vs_output(model, SELECTED, X_test, hist_input)
    shapley_bottleneck_vs_output(model, SELECTED, X_test, hist_bn)
    shapley_input_vs_bottleneck(model, SELECTED, X_test, hist_input)
    shapley_bottleneck_vs_class(model, SELECTED, X_test, hist_bn)
    shapley_input_vs_class(model, SELECTED, X_test, hist_input)


if __name__ == "__main__":
    main()
