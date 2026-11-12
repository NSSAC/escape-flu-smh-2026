#!/bin/bash
## One shard of an experiment's particles per array task. Submitted by `flu_calib.py submit`, which sets
## EXP, NSHARDS, the array range and the log path. Run `flu_calib.py prepare EXP` first (the build and
## input.h5 are shared by every task).
#SBATCH --account=bii_nssac
#SBATCH --partition=bii
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --mem=120G
#SBATCH --time=12:00:00

set -euo pipefail
. /etc/profile
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate escape
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-40}

REPO=$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd || echo /project/bii_nssac/biocomplexity/SMH/escape-flu-smh)
[[ -f "$REPO/scripts/calib/flu_calib.py" ]] || REPO=/project/bii_nssac/biocomplexity/SMH/escape-flu-smh
exec python "$REPO/scripts/calib/flu_calib.py" run "$EXP" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
