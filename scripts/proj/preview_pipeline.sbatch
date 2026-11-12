#!/bin/bash
## A whole production preview for a set of states as ONE unattended driver job: round 1 (Sobol explore), the
## next-season candidates, the screen pass, the early-season selection, the full scenario-B pass, the guard,
## scenarios A C D for the kept members, and the analyses. Each step is the foreground form of its command
## (sw_calib.py explore, flu_proj.py run: they host the task server and submit their own pilot jobs), so the next
## step starts when one finishes. Every step resumes (finished runs are skipped, members/selections are
## deterministic in the experiment's seed), so resubmitting the same job after a failure continues it.
##
##   sbatch --account=bii_nssac --partition=bii --qos=bii-half --time=2-00:00:00 --job-name=flu-preview \
##       --output=/scratch/$USER/flu-round1-2026-27/sw/drivers/%x-%j.out \
##       scripts/proj/preview_pipeline.sbatch "prod2025o_NM prod2025o_WI prod2025o_LA" preview-v3 \
##       "prod2025o_VA prod2025o_WA prod2025o_MA"
##
##   $1  experiments to run through every step
##   $2  analysis tag (flu-round1-2026-27/analysis/<tag>/)
##   $3  optional: experiments already through the guard; they get only the A C D step (their missing runs)
##       and join the analyses
##   env QOS (the pilot jobs', default bii-half), JOBS (48), POINTS (64), TOP (8), PER_FIT (16), SCREEN_N (90),
##       ESS (30), KEEP_N (30)
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=16G

. /etc/profile
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate "${FLU_SW_ENV:-escape-sw}"
set -euo pipefail

RUN=($1)
TAG=$2
ALSO=(${3:-})
QOS=${QOS:-bii-half}
JOBS=${JOBS:-48}
REPO=/project/bii_nssac/biocomplexity/SMH/escape-flu-smh
cd "$REPO"
PROJ="python -u scripts/proj/flu_proj.py"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "round 1: explore ${RUN[*]}"
python -u scripts/calib/sw_calib.py explore "${RUN[@]}" --points "${POINTS:-64}" --jobs "$JOBS" --qos "$QOS"

step "candidates: top ${TOP:-8} fits x ${PER_FIT:-16} next-season draws"
$PROJ members "${RUN[@]}" --design sobol --top "${TOP:-8}" --per-fit "${PER_FIT:-16}"

step "screen pass (cut a week after the origin)"
$PROJ run "${RUN[@]}" --mode screen --scenarios B --jobs "$JOBS" --qos "$QOS"

step "early-season selection: ${SCREEN_N:-90} members at ESS ${ESS:-30}"
$PROJ select "${RUN[@]}" --mode screen --weights early --n "${SCREEN_N:-90}" --ess "${ESS:-30}"

step "full scenario-B pass of the screened members"
$PROJ run "${RUN[@]}" --mode restart --selected --scenarios B --jobs "$JOBS" --qos "$QOS"

step "guard: keep ${KEEP_N:-30} against the past seasons"
$PROJ select "${RUN[@]}" --mode restart --weights guard --n "${KEEP_N:-30}"

step "scenarios A C D for the kept members of ${RUN[*]} ${ALSO[*]}"
$PROJ run "${RUN[@]}" "${ALSO[@]}" --mode restart --selected --scenarios A C D --jobs "$JOBS" --qos "$QOS"

step "analyses -> analysis/$TAG"
cd ../flu-round1-2026-27/notebooks
python -u run_analyses.py "${ALSO[@]}" "${RUN[@]}" --tag "$TAG" --us always --refresh-truth
step "done"
