#!/bin/bash
## The long-lived driver for state_pipeline.py (`state_pipeline.py submit run ...` submits this with the account,
## partition, time and log path). It hosts the ds-service server, submits the pilot jobs and runs the per-state
## analyses (one at a time), hence more memory than flu_proj.sbatch. Arguments are passed through.
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G

. /etc/profile
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate "${FLU_SW_ENV:-escape-sw}"
set -euo pipefail

REPO=/project/bii_nssac/biocomplexity/SMH/escape-flu-smh
cd "$REPO"
exec python -u scripts/proj/state_pipeline.py "$@"
