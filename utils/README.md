Execute from `path/to/TSAutoencoder`.

Perform the tuning of the model for a range of alphas or bottleneck neurons,

```
sbatch -p medium utils/run.sh tuning.py
```

For each alpha or bottleneck neurons repeat the training several times with different seeds to obtain the mean and standard deviation of the metrics,

```
sbatch -p medium --depend=<tuning_id> utils/run.sh experiments/exp2-acc_cor.py
```
