import random
import torch
import numpy as np

from models.CAE import CAE
from train import Trainer
from dataloader import ElectricDevices, normalize

import hydra
from hydra.experimental import initialize, compose

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import argparse
import json

random.seed(0) # Seed for the hyperparameter selection
np.random.seed(0)
torch.manual_seed(0)

def objective(config, data, cfg, checkpoint_dir=None):
    random.seed(1) # Seed for the training
    np.random.seed(1)
    torch.manual_seed(1)
    
    data_train, data_valid1, data_valid2 = data
    # HYPERPARAMETER SETTING
    cfg.model.lmd = config["lmd"]
    cfg.train.lr = config["lr"]
    cfg.train.early_stopping_rounds = config["early_stopping_rounds"]
    # END
    model = CAE(cfg.model, num_classes=7)
    trainer = Trainer(cfg.train)
    trainer.fit(model, data_train, data_valid1)
    loss = model.loss(data_valid2, do_reg=False).item()
    tune.report(loss=loss)

def main(
    num_alphas=21,
    num_samples=16,
    config={
        "lmd": tune.loguniform(1e-8, 1e-3),
        "lr": tune.loguniform(1e-6, 1e-1),
        "early_stopping_rounds": tune.randint(5, 30)
    }
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int)
    args = parser.parse_args()

    ray.init(include_dashboard=False, num_cpus=args.num_cpus)
    
    alphas = [float(f) for f in np.linspace(0, 1, num_alphas)]
    
    data_train_ori, data_valid_ori, _ = ElectricDevices()
    data_train, data_valid = normalize(data_train_ori), normalize(data_valid_ori)
    n_valid = data_valid.shape[0]
    data_valid1 = data_valid[:n_valid//2]
    data_valid2 = data_valid[n_valid//2:]
    data = (data_train, data_valid1, data_valid2)

    with initialize(config_path="configs"):
        cfg = compose(config_name="config")
    
    results = dict()
    for exp, alpha in enumerate(alphas):
        print("################################")
        print("################################")
        print(f"EXPERIMENT: alpha = {alpha}")
        print("################################")
        print("################################")

        cfg.model.alpha = alpha
        
        analysis = tune.run(
            lambda config, checkpoint_dir=None: objective(config, data, cfg, checkpoint_dir),
            name=f"alpha_{exp}",
            num_samples=num_samples,
            config=config,
            search_alg=HyperOptSearch(metric="loss", mode="min", random_state_seed=exp),
            verbose=2
        )
        best_config = analysis.get_best_config(metric="loss", mode="min")
        
        results.update({exp: {"alpha": alpha, "hyperparams": best_config}})
        with open("../../exp.json", "w") as f:
            json.dump(results, f, indent=4)

        print("Best config: ", best_config)
        print()
        print()

if __name__ == "__main__":
    main()
