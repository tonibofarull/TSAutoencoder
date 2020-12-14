import torch
from torch.utils.data import DataLoader
from torch import optim
import time

class Trainer():
    def __init__(self, cfg):
        self.lr = cfg.lr
        self.early_stopping_rounds = cfg.early_stopping_rounds
        self.iters = cfg.iters
        self.batch_size = cfg.batch_size
        self.shuffle = cfg.shuffle
        self.verbose = cfg.verbose

    def fit(self, model, X_train, X_valid):
        # Datasets and optimizer
        trainloader = DataLoader(X_train, batch_size=self.batch_size, shuffle=self.shuffle)
        validloader = DataLoader(X_valid, batch_size=self.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Training loop
        tini = time.time()
        train_losses, valid_losses = [], []
        min_loss = (float("inf"),-1) # used for early stopping
        for epoch in range(1, self.iters+1):
            train_running_loss = 0
            model.train()
            for batch in trainloader:
                train_loss = model.loss(batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item()

            train_running_loss /= len(trainloader)
            model.eval()
            with torch.no_grad():
                valid_running_loss = 0
                for batch in validloader:
                    valid_running_loss += model.loss(batch).item()
                valid_running_loss /= len(validloader)

            train_losses.append(train_running_loss)
            valid_losses.append(valid_running_loss)

            if valid_running_loss < min_loss[0]:
                min_loss = (valid_running_loss, epoch)

            end_train = self.early_stopping_rounds is not None and epoch > self.early_stopping_rounds
            end_train = end_train and epoch-min_loss[1] >= self.early_stopping_rounds

            if self.verbose and (epoch%10 == 0 or end_train):
                print(f"EPOCH {epoch} train loss: {train_running_loss}, valid loss: {valid_running_loss}")
                print(f"epochs without improvement: {epoch-min_loss[1]}")
                print()

            if end_train:
                break

        print(f"Training Finished in {(time.time()-tini)}s")
        return train_losses, valid_losses
