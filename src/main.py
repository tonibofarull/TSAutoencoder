import argparse
import os
import random

import hydra
import numpy as np
import torch
from dataloader import ElectricDevices
from hydra.experimental import compose
from hydra.experimental import initialize_config_dir
from interpretability import get_hist
from interpretability import global_interpretability
from interpretability import input_max_neuron
from interpretability import neuron_max_class
from interpretability import shapley_bottleneck_vs_class
from interpretability import shapley_bottleneck_vs_output
from interpretability import shapley_input_vs_bottleneck
from interpretability import shapley_input_vs_class
from interpretability import shapley_input_vs_output
from models.CAE import CAE
from utils import accuracy
from utils import data_bottleneck_exploration
from utils import data_input_exploration
from utils import get_predictions
from utils import observation_reconstruction
from utils import reconstruction


torch.manual_seed(4444)
np.random.seed(4444)
random.seed(4444)

SELECTED = [3279, 1156, 7419, 5046, 3323, 6485, 5497]
# SELECTED = [
#     np.random.choice([i for i, x in enumerate(y_test) if int(x) == j])
#     for j in range(cfg.model.num_classes)
# ]


def evaluate_model(model, X_test, y_test):
    X_testp, y_testp = get_predictions(model, X_test)

    reconstruction(X_test, X_testp)
    accuracy(y_test, y_testp)
    observation_reconstruction(SELECTED, X_test, X_testp, y_test, y_testp)


def data_exploration(model, X_train):
    data_input_exploration(X_train)
    data_bottleneck_exploration(model, X_train)

    # input_max_neuron(model)
    # neuron_max_class(model)


def interpretability(model, X_train, X_test, length, bottleneck_nn):
    global_interpretability(model)

    hist_input = [get_hist(X_train[:, 0, i]) for i in range(length)]
    aux = model.encoder(X_train, False).detach().numpy()
    hist_bn = [get_hist(aux[:, i]) for i in range(bottleneck_nn)]

    shapley_input_vs_output(model, SELECTED, X_test, hist_input)
    shapley_bottleneck_vs_output(model, SELECTED, X_test, hist_bn)
    shapley_input_vs_bottleneck(model, SELECTED, X_test, hist_input)
    shapley_bottleneck_vs_class(model, SELECTED, X_test, hist_bn)
    shapley_input_vs_class(model, SELECTED, X_test, hist_input)


def feature_visualization(model):
    input_max_neuron(model)
    neuron_max_class(model)


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

    evaluate_model(model, X_test, y_test)
    data_exploration(model, X_train)
    interpretability(model, X_train, X_test, cfg.model.length, cfg.model.bottleneck_nn)
    feature_visualization(model)


if __name__ == "__main__":
    main()
