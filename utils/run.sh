#!/bin/bash
#SBATCH --cpus-per-task=12

cd src

# export PYTHONPATH=${PYTHONPATH}:/path/to/TSAutoencoder/src
export PYTHONPATH=${PYTHONPATH}:/home/usuaris/tbofarull/TSAutoencoder/src

time srun python -u $1 --num_cpus 12

echo "--- DONE ---"
