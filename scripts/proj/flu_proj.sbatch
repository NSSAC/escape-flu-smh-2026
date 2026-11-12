#!/bin/bash
## The long-lived driver for flu_proj.py (`flu_proj.py submit run ...` submits this with the account,
## partition, time and log path). It hosts the ds-service server and submits the pilot jobs; it needs no
## simulation resources itself. Arguments are passed through to flu_proj.py.
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=16G

. /etc/profile
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate "${FLU_SW_ENV:-escape-sw}"
set -euo pipefail

REPO=/project/bii_nssac/biocomplexity/SMH/escape-flu-smh
cd "$REPO"
exec python -u scripts/proj/flu_proj.py "$@"
