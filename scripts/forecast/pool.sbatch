#!/bin/bash
## A forecasting study's particle pool in one driver job: screen runs -> loose pre-screen -> full runs of the kept.
## The driver hosts ds-service and submits the pilot jobs (flu_proj.py run); it needs no simulation resources.
## members.csv must exist first (flu_proj.py members ... --design sobol). Every step resumes: resubmitting the
## same command skips finished runs, and the pre-screen is redone only if restart/selected.csv is missing.
##
##   source /project/bii_nssac/biocomplexity/SMH/flu-forecasting-FluTURE-2026-27/env.sh
##   mkdir -p /scratch/$USER/flu-forecasting-FluTURE-2026-27/sw/drivers
##   sbatch --account=bii_nssac --partition=bii --qos=bii-half --time=2-00:00:00 \
##          --output=/scratch/$USER/flu-forecasting-FluTURE-2026-27/sw/drivers/%x-%j.out \
##          $FLU_REPO/scripts/forecast/pool.sbatch fc2627_VA fc2627_MA fc2627_WA
##
## Pilot sizing: RUN_ARGS (default "--jobs 54 --qos bii-half"; bii's default QOS caps a user at 400 CPUs);
## add e.g. "--extra-partition bii-largemem --extra-jobs 30" for a second group. PRESCREEN_ARGS (default
## "--n 384 --band 4") are fc.py prescreen's options.
#SBATCH --job-name=fc-pool
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=16G

. /etc/profile
STUDY=${FLU_ROUND:-/project/bii_nssac/biocomplexity/SMH/flu-forecasting-FluTURE-2026-27}
source "$STUDY/env.sh"
set -euo pipefail
cd "$FLU_REPO"

RUN_ARGS=${RUN_ARGS:---jobs 54 --qos bii-half}
PRESCREEN_ARGS=${PRESCREEN_ARGS:---n 384 --band 4}
echo "study $FLU_ROUND  scratch $FLU_SCRATCH  experiments $*  run args: $RUN_ARGS"

# 1. screen runs: every member, scenario B, cut at production.screen_end
python -u scripts/proj/flu_proj.py run "$@" --mode screen --scenarios B $RUN_ARGS

# 2. loose pre-screen on NHSN to production.data_end -> proj_<exp>/restart/selected.csv
todo=()
for e in "$@"; do
  [ -f "$FLU_ROUND/proj_$e/restart/selected.csv" ] || todo+=("$e")
done
if [ ${#todo[@]} -gt 0 ]; then
  python -u scripts/forecast/fc.py prescreen "${todo[@]}" $PRESCREEN_ARGS
fi

# 3. the kept members in full, to the study's proj_end
python -u scripts/proj/flu_proj.py run "$@" --mode restart --selected --scenarios B $RUN_ARGS
python -u scripts/forecast/fc.py status
