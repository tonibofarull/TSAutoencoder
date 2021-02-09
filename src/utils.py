from captum.attr import GradientShap
from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import ShapleyValueSampling
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import torch
import numpy as np

def get_shapley_values(model, inp, target, n_samples=50, batch=50):
    def model_wrapper(x):
        return model(x)[0]

    inp.requires_grad_()

    baselines = tuple(torch.rand((1,1,96)) for _ in range(100))

    inter = ShapleyValueSampling(model_wrapper)
    attr = inter.attribute(inp, target=(0,target), 
        baselines=baselines,
        n_samples=n_samples, 
        perturbations_per_eval=batch
    )

    return attr.detach().numpy()
