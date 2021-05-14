import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import random

from models.CAE import CAE
from train import Trainer
from interpretability import *
from dataloader import ElectricDevices, normalize
from utils import baseline

from pingouin import distance_corr # Szekely and Rizzo

import hydra
from hydra.experimental import initialize, initialize_config_dir, compose

torch.manual_seed(4444)
np.random.seed(4444)
random.seed(4444)

from utils import shapley_input_vs_output
from utils import shapley_bottleneck_vs_output
from utils import shapley_input_vs_bottleneck
from utils import shapley_bottleneck_vs_class
from utils import shapley_input_vs_class
from utils import shapley_bottleneck_vs_output

SELECTED = [3279, 1156, 7419, 5046, 3323, 6485,5497]

# Autoencoder

def predict(model, data_test):
    X_test, y_test = data_test[:,:,:-1], data_test[:,:,-1].numpy()
    X_testp, outclass_testp, bn = model(X_test, apply_noise=False)
    X_testp = X_testp.detach().numpy()
    probs_testp = model.classifier.get_probs(outclass_testp)
    y_testp = torch.argmax(probs_testp, dim=1).detach().numpy()
    return X_testp, y_testp

def reconstruction(X_test, X_testp):
    cors = [distance_corr(X_testp[i,0], X_test[i,0].detach().numpy(), n_boot=None) for i in range(X_test.shape[0])]
    print("Distance Correlation avg and std:", np.mean(cors), np.std(cors))
    print("NRMSE:", (torch.sqrt(torch.mean(torch.square(X_test-X_testp)))/(torch.max(X_test)-torch.min(X_test))).item())
    print()

def accuracy(y_test, y_testp):
    cm = confusion_matrix(y_test, y_testp)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    print("Accuracy:", np.sum(np.diag(cm))/np.sum(cm))
    plt.show()

def observation_reconstruction(selected, X_test, X_testp, y_test, y_testp):
    fig, axs = plt.subplots(nrows=2, ncols=len(selected), figsize=(25,5), constrained_layout=True)
    for i, x in enumerate(selected):
        axs[0,i].set_title(f"Real class: {int(y_test[x][0])}")
        axs[0,i].plot(X_test[x,0])
        axs[0,i].axis("off")
        axs[0,i].set_ylim((0,1))

        axs[1,i].set_title(f"Predicted class: {int(y_testp[x])}")
        axs[1,i].plot(X_testp[x,0])
        axs[1,i].axis("off")
        axs[1,i].set_ylim((0,1))

        print("cor:", distance_corr(X_testp[x,0], X_test[x,0], n_boot=None))
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
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(25,10), constrained_layout=True)
    for i in range(7): #selected):
        a1 = feature_visualization(model, i, 1)
        axs.flat[i].plot(a1)
        axs.flat[i].set_title(f"Representant class {i}")
        axs.flat[i].set_ylim(-0.05, 1.05)

# Interpretability

def global_interpretability(model):
    num_filter = model.k*model.M
    w_per_filter = model.length
    num_neurons = model.bottleneck_nn
    M = model.M

    w = np.array([[torch.mean(torch.abs(model.encoder.fc_conv_bn.weight[j,i*w_per_filter:(i+1)*w_per_filter])).item() for i in range(num_filter)] for j in range(num_neurons)])
    x_axis_labels = [f"{i}-d:{model.dilation[i//M]}" for i in range(w.shape[1])]

    _ = sns.heatmap(w, xticklabels=x_axis_labels, cmap="gray", vmin=0)
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

    with initialize_config_dir(config_dir=os.path.abspath("../src/configs")):
        cfg = compose(config_name="config")

    data_train_ori, data_valid_ori, data_test_ori = ElectricDevices()
    data_train, data_valid, data_test = normalize(data_train_ori), normalize(data_valid_ori), normalize(data_test_ori)
    X_train, y_train = data_train[:,:,:-1], data_train[:,:,-1]
    X_valid, y_valid = data_valid[:,:,:-1], data_valid[:,:,-1]
    X_test, y_test = data_test[:,:,:-1], data_test[:,:,-1]

    model = CAE(cfg.model, num_classes=7)
    # torch.save(model.state_dict(), "../weights/mod.pth")
    model.load_state_dict(torch.load("../weights/mod.pth"))

    X_testp, y_testp = predict(model, data_test)

    reconstruction(X_test, X_testp)
    accuracy(y_test, y_testp)

    # selected = np.random.choice(range(1000), 7)

    observation_reconstruction(SELECTED, X_test, X_testp, y_test, y_testp)

    # INTERPRETABILITY
    global_interpretability(model)

    hist_input = [get_hist(X_train[:,0,i]) for i in range(96)]
    aux = model.encoder(X_train, False).detach().numpy()
    hist_bn = [get_hist(aux[:,i]) for i in range(24)]

    shapley_input_vs_output(model, SELECTED, X_test, hist_input)
    shapley_bottleneck_vs_output(model, SELECTED, X_test, hist_bn)
    shapley_input_vs_bottleneck(model, SELECTED, X_test, hist_input)
    shapley_bottleneck_vs_class(model, SELECTED, X_test, hist_bn)
    shapley_input_vs_class(model, SELECTED, X_test, hist_input)

if __name__ == "__main__":
    main()
