#!/bin/bash
## The long-lived driver for sw_calib.py (`sw_calib.py submit explore|optimize ...` submits this with the
## account, partition, time and log path). It hosts the ds-service server and submits the pilot jobs; it
## needs no simulation resources itself. Arguments are passed through to sw_calib.py.
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=16G

. /etc/profile
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate "${FLU_SW_ENV:-escape-sw}"
set -euo pipefail

REPO=/project/bii_nssac/biocomplexity/SMH/escape-flu-smh
cd "$REPO"
exec python -u scripts/calib/sw_calib.py "$@"
