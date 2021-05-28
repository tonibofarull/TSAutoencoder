Execute from `path/to/TSAutoencoder`.

```
sbatch -p medium utils/run.sh tuning.py
```

```
sbatch -p medium --depend=<tuning_id> utils/run.sh experiments/exp2-acc_cor.py
```

```
sbatch -p medium utils/run.sh experiments/exp2-acc_cor.py
```
