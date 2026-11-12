# escape-flu-smh

The ESCAPE-ABM influenza model and its calibration and projection drivers, built for **SMH Flu Round 1 2026-27**
(hub: [midas-network/flu-scenario-modeling-hub](https://github.com/midas-network/flu-scenario-modeling-hub)).
The submission is `UVA-escape_flu`: round 1 was hub PR #57 (prod-60-v2), and the Nov 2026 update was prod-v5.

Start with [AGENT_HANDOFF.md](AGENT_HANDOFF.md) to continue the work or set up a later round. It covers the
history, decisions, pitfalls and open items. [docs/flu.md](docs/flu.md) is the detailed lab notebook of the model
and the calibration, through prod-60-v2.

## Two directories

| | what | in git |
|---|---|---|
| `SMH/escape-flu-smh/` (this repo) | model (`models/*.esl`), driver package `py-modules/flu_smh`, experiment JSONs, SLURM drivers | yes |
| `SMH/flu-round1-2026-27/` (the workspace) | hub and surveillance data, person files, analysis code (`notebooks/*.py`), the hub submission (`submission/`), and every run's outputs (`calib_<exp>/`, `proj_<exp>/`, `analysis/`) | **no** (see AGENT_HANDOFF.md) |

The drivers find the workspace through `FLU_ROUND`, which defaults to
`/project/bii_nssac/biocomplexity/SMH/flu-round1-2026-27`. They find scratch through `FLU_SCRATCH`, which defaults
to `/scratch/$USER/flu-round1-2026-27/calib`. The workspace scripts find this repo as `../escape-flu-smh`.

## Layout

| path | contents |
|---|---|
| `models/flu-r1-2026-27-v4.esl` | **current model** (used by prodv3/v3g/v5). It has three subtype contagions (`flu_A_H1N1`, `flu_A_H3N2`, `flu_B`); subtype × age hospitalization and death; A↔A cross-protection; config seasonality with a next-season block; long-term immunity; an H1N1 seeding pulse (v3); and a hub-matched vaccine (`vax_escape_per_day`, `vax_symp_ve`; v4) |
| `models/flu-r1-2026-27{,-v2,-v3}.esl` | earlier versions. They are kept unchanged because experiments point at them by name and running workers re-render them |
| `models/covid19-r20.esl`, `model.esl`, `spok24.esl`, `vscode-esl/` | inherited from COVID Round 20 and the cryptic round; reference only |
| `py-modules/flu_smh/` | `experiment.py` (experiment JSON, paths, iteration registry), `schedule.py` (vaccination schedules), `sim.py` (build, run, extract), `calib.py` (priors, draws, NHSN likelihood, scoring), `sw.py` (slurm-workflows objective), `proj.py` (members, restart mode, screen, early selection, guards, top-ups), `upscale.py` (severity and death replicates, all-deaths scaling) |
| `scripts/calib/` | `sw_calib.py` (multi-state explore and optimize on pilot jobs), `seed_from.py`, `flu_calib.py` (the first single-node driver) |
| `scripts/proj/` | `flu_proj.py` (members, run, select), `state_pipeline.py` (per-state chain from prepare to analysis), `make_seedx.py`, `make_prodv3.py` (v3, v3g, v4), `v5_pipeline.py` (top-up), sbatch wrappers |
| `experiments/` | one JSON per experiment (`<family>_<ST>.json`) and `iterations.json`, the registry of iterations it01 to it12 that maps families to report tags |
| `docs/flu.md` | model and calibration notes, round by round |
| `docs/calibration_and_projection.md`, `docs/round20.md` | the Round 20 process (background) |
| `py-modules/escape_covid19_smh/`, `scripts/*.py` (top level) | Round 20 / SPOK-24 drivers. Reference only: their paths point at covid19-round20 and other users' homes |

## Environment

```bash
module load gcc/14.2.0                      # in the same shell as the build, or libhdf5_cpp fails to link
eval "$($HOME/conda/bin/conda shell.bash hook)"; conda activate escape-sw   # escape-abm + slurm-workflows 5.1 + optimize
cd escape-flu-smh                           # experiment globs ($E=...) resolve against experiments/ here
```

## Commands

Most commands take experiment names or globs. The full chain is in AGENT_HANDOFF.md §5.

```bash
python scripts/calib/sw_calib.py submit explore <exps> ...             # round-1 calibration on pilot jobs
python scripts/proj/state_pipeline.py submit run <exps> --tag <tag> ...  # per-state production chain
python scripts/proj/flu_proj.py status <exps> --mode restart
python ../flu-round1-2026-27/notebooks/run_analyses.py <exps> --tag <tag> --upscale-to 300 --all-deaths
python ../flu-round1-2026-27/notebooks/smh_submission.py <exps> --selected --upscale-to 300 --deaths --all-deaths \
    --age-groups --quantiles --national always
```

Every SLURM submission prints or writes an sbatch command. Queue limits: the partition `bii` caps a user at 400
CPUs, so wide runs use `--qos bii-half` or the partition `bii-largemem`.

## Lineage

[NSSAC/escape-covid19-smh](https://github.com/NSSAC/escape-covid19-smh) (COVID Round 20) →
[NSSAC/escape-cryptic-smh](https://github.com/NSSAC/escape-cryptic-smh) (cryptic round, SPOK-24) → this repo. This
repo was copied without history and started its own git history in Nov 2026.
