#!/bin/bash
#SBATCH --job-name=explore-params
#SBATCH --output=explore-params-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --account=bii_nssac --qos=bii-unlimited
#SBATCH --partition=bii --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --mem=90G

. /etc/profile

module load gcc/14.2.0
eval "$(/home/pb5gj/miniforge3/bin/conda shell.bash hook)"
conda activate hp13

exec python explore_params.py
