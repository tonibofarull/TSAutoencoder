from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from functional import *
import time

def VAE_loss(model, batch, lmd=0.1):
    pred, mean, logvar = model.forward(batch, is_train=True)

    recon = torch.mean((pred-batch)**2)
    KL = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + torch.square(mean) - 1.0 - logvar, dim=0))

    return recon + lmd*KL #+ my_l1(model)

def AE_loss(model, batch):
    pred = model(batch)
    return mse_loss(pred,batch) + my_l1(model)

def train(model, X_train, X_valid, iters, early_stopping_rounds=10, verbose=True):
    trainloader = DataLoader(X_train, batch_size=32, shuffle=True)
    validloader = DataLoader(X_valid, batch_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = VAE_loss

    tini = time.time()
    train_losses, valid_losses = [], []
    min_loss = (float("inf"),-1) # used for early stopping
    for epoch in range(1, iters+1):
        train_running_loss = 0
        model.train()
        for batch in trainloader:
            train_loss = loss(model, batch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_running_loss += train_loss.item()

        train_running_loss /= len(trainloader)
        model.eval()
        with torch.no_grad():
            valid_running_loss = 0
            for batch in validloader:
                valid_running_loss += (loss(model, batch)).item()
            valid_running_loss /= len(validloader)

        train_losses.append(train_running_loss)
        valid_losses.append(valid_running_loss)

        if valid_running_loss < min_loss[0]:
            min_loss = (valid_running_loss, epoch)

        if verbose and epoch%10 == 0:
            print(f"EPOCH {epoch} train loss: {train_running_loss}, valid loss: {valid_running_loss}")
            print(f"epochs without improvement: {epoch-min_loss[1]}")
            print()

        if early_stopping_rounds is not None and epoch > early_stopping_rounds and epoch-min_loss[1] >= early_stopping_rounds:
            break

    print(f"Training Finished in {(time.time()-tini)}s")
    return train_losses, valid_losses
