from captum.attr import GradientShap
from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import ShapleyValueSampling
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from captum.attr import LayerFeatureAblation, LayerConductance
from captum.attr import NeuronFeatureAblation
from captum.attr import GradientShap
from captum.attr import ShapleyValueSampling

import torch
import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

def get_shapley_values(inp, model, targets, baselines, n_samples=64, perturbations_per_eval=64):
    inp = inp.reshape((-1,1,96))
    input_feature = ShapleyValueSampling(lambda x: model(x)[0])
    input_attrs = []
    for i in targets:
        print(i, end=" ")
        attr = input_feature.attribute(inp, target=(0,i), 
            baselines=baselines,
            n_samples=n_samples, 
            perturbations_per_eval=perturbations_per_eval
        )
        attr = np.mean(attr.detach().numpy(), axis=0).flatten() # smoothgrad: mean of attributions
        input_attrs.append(attr)
    return input_attrs

def get_layer_attrs(inp, model, layer, targets, baselines=None):
    inp = inp.reshape((-1,1,96))
    layer_feature = LayerFeatureAblation(lambda x : model(x)[0], layer)
    layer_attrs = []
    for i in targets:
        attr = layer_feature.attribute(inp, target=(0,i), layer_baselines=baselines)
        attr = attr.detach().numpy().flatten()
        layer_attrs.append(attr)
    return layer_attrs

def get_neuron_attrs(inp, model, layer, targets, baselines=None):
    inp = inp.reshape((-1,1,96))
    neuron_feature = NeuronFeatureAblation(lambda x : model(x)[0], layer)
    neuron_attrs = []
    for i in targets:
        attr = neuron_feature.attribute(inp, neuron_selector=i, baselines=baselines)
        attr = attr.detach().numpy().flatten()
        neuron_attrs.append(attr)
    return neuron_attrs

# ...

def plot_shapley_ts(inp, sv_vec, filename):
    ylim = (np.min(sv_vec),1)
    for i in range(96):
        plt.plot(sv_vec[i].flatten())
        plt.plot(inp.detach().numpy().flatten())
        plt.title(f"Position {i}")
        plt.ylim(ylim)
        plt.axvline(x=i, c="red")
        plt.savefig(f"../plots/{filename}.png")
        plt.close()

def baseline(data_train, data_valid, data_test):
    X_train, y_train = data_train[:,0,:-1], data_train[:,0,-1].numpy()
    X_valid, y_valid = data_valid[:,0,:-1], data_valid[:,0,-1].numpy()
    X_test, y_test = data_test[:,0,:-1], data_test[:,0,-1].numpy()
    y_train.astype(int)
    y_valid.astype(int)
    y_test.astype(int)

    best_acc = 0
    best_k = 0
    for i in range(1,50):

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, y_train)

        y_validp = neigh.predict(X_valid)
        y_validp.astype(int)

        acc = np.sum(y_validp == y_valid)/len(y_valid)
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
    print("Test acc:", np.sum(np.diag(cm))/np.sum(cm))
