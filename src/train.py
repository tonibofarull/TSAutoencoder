import torch
from torch.utils.data import DataLoader
from torch import optim
import time
from copy import deepcopy

class Trainer():
    def __init__(self, cfg,
        restore_best_weights = False
    ):
        self.lr = cfg.lr
        self.early_stopping_rounds = cfg.early_stopping_rounds
        self.iters = cfg.iters
        self.batch_size = cfg.batch_size
        self.shuffle = cfg.shuffle
        self.verbose = cfg.verbose

        self.restore_best_weights = restore_best_weights

    def fit(self, model, X_train, X_valid):
        # Datasets and optimizer
        trainloader = DataLoader(X_train, batch_size=self.batch_size, shuffle=self.shuffle)
        validloader = DataLoader(X_valid, batch_size=self.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Training loop
        tini = time.time()
        train_losses, valid_losses = [], []
        early_stop = {
            'rounds': self.early_stopping_rounds,
            'best_cost': float('inf'),
            'best_round': None,
            'best_model': None
        }
        
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

            if valid_running_loss < early_stop['best_cost']:
                early_stop['best_cost'] = valid_running_loss
                early_stop['best_round'] = epoch
                if self.restore_best_weights:
                    early_stop['best_model'] = deepcopy(model.state_dict()) # we must do a deep copy

            end_train = early_stop['rounds'] is not None and epoch > early_stop['rounds']
            end_train = end_train and epoch-early_stop['best_round'] >= early_stop['rounds']

            if self.verbose and (epoch%10 == 0 or end_train):
                print(f"EPOCH {epoch} train loss: {train_running_loss}, valid loss: {valid_running_loss}")
                print(f"epochs without improvement: {epoch-early_stop['best_round']}")
                print()

            if end_train:
                break

        # Final update of the model

        if self.restore_best_weights:
            model.load_state_dict(early_stop['best_model'])

        print(f"Training Finished in {(time.time()-tini)}s")
        return train_losses, valid_losses
