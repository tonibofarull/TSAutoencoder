import torch
import numpy as np
from torch import optim

def get_hist(x):
    x = x.reshape(-1,1)
    n = x.shape[0]
    counts, bins = np.histogram(x, bins="auto") # TODO: check auto

    counts = n-counts
    probs = counts/sum(counts)
    return bins, probs

def sample_from_hist(hist, size=1):
    """
    Select bin with probability and uniformly selects and element of the bin
    """
    bins, probs = hist
    As = np.random.choice(a=range(len(probs)), p=probs, replace=True, size=size)
    elems = [np.random.uniform(bins[a], bins[a+1]) for a in As]
    return np.array(elems)

# Shapley Values

def shapley_sampling(x, model, feature, n_batches=1, batch_size=64, histograms=None):
    """
    Importance of input with respect the output
    """
    length = x.shape[-1]
    sv = torch.zeros(length)

    x = x.reshape(1, length).repeat(batch_size, 1)
    for _ in range(n_batches):
        # y = torch.rand((batch_size, length))
        y = torch.zeros((batch_size, length))

        for i in range(96):
            y[:,i] = torch.tensor(sample_from_hist(histograms[i], size=batch_size))
        # END OF SAMPLING
        
        O = np.array([np.random.permutation(length) for _ in range(batch_size)])
        idx = np.where(O == feature)
        Os = [O[i,:j] for i, j in zip(idx[0], idx[1])]

        sel = torch.zeros((batch_size,length), dtype=torch.bool)
        sel[np.concatenate([np.repeat(i,len(Os[i])) for i in range(batch_size)]), np.concatenate(Os)] = True

        x2 = torch.where(sel, x, y)
        x1 = x2.clone()
        x1[:,feature] = x[:,feature]

        x1 = x1.reshape(-1,1,length)
        x2 = x2.reshape(-1,1,length)
        with torch.no_grad():
            v1 = model(x1, False)[0][:,0]
            v2 = model(x2, False)[0][:,0]
        sv += torch.sum(v1 - v2, axis=0)

    sv /= n_batches*batch_size
    return sv.numpy()

def shapley_sampling_bottleneck_output(x, model, feature, baselines, n_batches=1, batch_size=64):
    """
    Importance of neuron with respect the output
    """
    sv = torch.zeros(x.shape[-1])
    x = model.encoder(x.reshape(1,1,96), False) # bottleneck
    length = x.shape[-1]
    for _ in range(n_batches):
        if baselines is None:
            y = torch.zeros((batch_size, length)) # changed
        else:
            y = torch.tensor(baselines).reshape(1,24).repeat((batch_size,1))
        O = np.array([np.random.permutation(length) for _ in range(batch_size)])
        idx = np.where(O == feature)
        Os = [O[i,:j] for i, j in zip(idx[0], idx[1])]

        sel = torch.zeros((batch_size,length), dtype=torch.bool)
        sel[np.concatenate([np.repeat(i,len(Os[i])) for i in range(batch_size)]), np.concatenate(Os)] = True

        x2 = torch.where(sel, x, y)
        x1 = x2.clone()
        x1[:,feature] = x[:,feature]

        x1 = x1.reshape(-1,1,length)
        x2 = x2.reshape(-1,1,length)
        with torch.no_grad():
            v1 = model.decoder(x1)[:,0]
            v2 = model.decoder(x2)[:,0]
        sv += torch.sum(v1 - v2, axis=0)

    sv /= n_batches*batch_size
    return sv.numpy()

def shapley_sampling_bottleneck_class(x, model, feature, baselines, n_batches=1, batch_size=64):
    sv = torch.zeros(7)
    x = model.encoder(x.reshape(1,1,96), False) # bottleneck
    length = x.shape[-1]
    for _ in range(n_batches):
        if baselines is None:
            y = torch.zeros((batch_size, length)) # changed
        else:
            y = torch.tensor(baselines).reshape(1,24).repeat((batch_size,1))
        O = np.array([np.random.permutation(length) for _ in range(batch_size)])
        idx = np.where(O == feature)
        Os = [O[i,:j] for i, j in zip(idx[0], idx[1])]

        sel = torch.zeros((batch_size,length), dtype=torch.bool)
        sel[np.concatenate([np.repeat(i,len(Os[i])) for i in range(batch_size)]), np.concatenate(Os)] = True

        x2 = torch.where(sel, x, y)
        x1 = x2.clone()
        x1[:,feature] = x[:,feature]

        with torch.no_grad():
            v1 = model.classifier.get_probs(model.classifier(x1))
            v2 = model.classifier.get_probs(model.classifier(x2))
        sv += torch.sum(v1 - v2, axis=0)

    sv /= n_batches*batch_size
    return sv.numpy()

def shapley_sampling_input_bottleneck(x, model, feature, n_batches=1, batch_size=64):
    length = x.shape[-1]
    sv = torch.zeros(model.bottleneck_nn)

    x = x.reshape(1, length).repeat(batch_size, 1)
    for _ in range(n_batches):
        y = torch.rand((batch_size, length))
        O = np.array([np.random.permutation(length) for _ in range(batch_size)])
        idx = np.where(O == feature)
        Os = [O[i,:j] for i, j in zip(idx[0], idx[1])]

        sel = torch.zeros((batch_size,length), dtype=torch.bool)
        sel[np.concatenate([np.repeat(i,len(Os[i])) for i in range(batch_size)]), np.concatenate(Os)] = True

        x2 = torch.where(sel, x, y)
        x1 = x2.clone()
        x1[:,feature] = x[:,feature]

        x1 = x1.reshape(-1,1,length)
        x2 = x2.reshape(-1,1,length)
        with torch.no_grad():
            v1 = model.encoder(x1, False)
            v2 = model.encoder(x2, False)
        sv += torch.sum(v1 - v2, axis=0)

    sv /= n_batches*batch_size
    return sv.numpy()

def shapley_sampling_input_class(x, model, feature, n_batches=1, batch_size=64):
    length = x.shape[-1]
    sv = torch.zeros(model.num_classes)

    x = x.reshape(1, length).repeat(batch_size, 1)
    for _ in range(n_batches):
        y = torch.rand((batch_size, length))
        O = np.array([np.random.permutation(length) for _ in range(batch_size)])
        idx = np.where(O == feature)
        Os = [O[i,:j] for i, j in zip(idx[0], idx[1])]

        sel = torch.zeros((batch_size,length), dtype=torch.bool)
        sel[np.concatenate([np.repeat(i,len(Os[i])) for i in range(batch_size)]), np.concatenate(Os)] = True

        x2 = torch.where(sel, x, y)
        x1 = x2.clone()
        x1[:,feature] = x[:,feature]

        x1 = x1.reshape(-1,1,length)
        x2 = x2.reshape(-1,1,length)
        with torch.no_grad():

            v1 = model.classifier.get_probs(model(x1, False)[1])
            v2 = model.classifier.get_probs(model(x2, False)[1])
        sv += torch.sum(v1 - v2, axis=0)

    sv /= n_batches*batch_size
    return sv.numpy()

# Feature Visualization

def feature_visualization(model, neuron):
    """
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """
    X = torch.rand((1,1,96), requires_grad=True)
    optimizer = optim.LBFGS([X])

    def closure():
        optimizer.zero_grad()
        model.zero_grad()
        _, _, bn = model(torch.sigmoid(X), apply_noise=False)
        y = -bn[0,neuron]
        y.backward()
        return y

    for i in range(100):
        optimizer.step(closure)
    return torch.sigmoid(X).detach().numpy().flatten()

def feature_visualization_class(model, clss):
    """"
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """
    X = torch.rand((1,1,96), requires_grad=True)
    optimizer = optim.LBFGS([X])

    def closure():
        optimizer.zero_grad()
        model.zero_grad()
        _, pred_class, _ = model(torch.sigmoid(X), apply_noise=False)
        y = -pred_class[0,clss]
        y.backward()
        return y

    for i in range(100):
        optimizer.step(closure)
    return torch.sigmoid(X).detach().numpy().flatten()