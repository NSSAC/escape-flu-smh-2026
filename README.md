# escape-flu-smh

An agent-based model of seasonal influenza in ESCAPE-ABM's ESL, and the drivers that calibrate it to
surveillance data and project vaccination scenarios on HPC.
- **Scale and structure:** state-scale synthetic populations, three co-circulating subtypes (A(H1N1), A(H3N2), B),
  and per-agent infection and vaccination histories.
- **Outputs:** paired scenario projections of infections, hospital admissions and deaths by subtype × age.

The model and pipeline were built for the Scenario Modeling Hub's **Flu Round 1 2026-27**
(submission `UVA-escape_flu`), but nothing in them is specific to SMH:
- a study is a set of experiment JSONs (calendar, fitting cutoff, locations, scenarios);
- the hub file is one export of the outputs.

The data, analysis and export code is in the companion pipeline repository (`SMH/flu-round1-2026-27`), whose
`docs/studies.md` describes how to set up another study.

To continue the work, start with [AGENT_HANDOFF.md](AGENT_HANDOFF.md): history, decisions, pitfalls, open items.

## Two repositories

| | what |
|---|---|
| this repo (`SMH/escape-flu-smh`) | model (`models/*.esl`), driver package `py-modules/flu_smh`, experiment JSONs, SLURM drivers |
| pipeline (`SMH/flu-round1-2026-27`) | surveillance data, person files (initial conditions), analyses, per-run analyses, the SMH export, and every run's outputs (`calib_<exp>/`, `proj_<exp>/`, `analysis/`, outputs not tracked) |

The drivers write outputs under `FLU_ROUND`, which defaults to `/project/bii_nssac/biocomplexity/SMH/flu-round1-2026-27`,
and build under `FLU_SCRATCH`, which defaults to `/scratch/$USER/flu-round1-2026-27/calib`.

## Layout

| path | contents |
|---|---|
| `models/flu-r1-2026-27-v4.esl` | **current model**. It has subtype contagions; subtype × age hospitalization and death; A↔A cross-protection; config seasonality with a next-season block; long-term immunity; an H1N1 seeding pulse (v3); and a vaccine acting on infection, symptoms and hospitalization (v4; [docs/vaccine.md](docs/vaccine.md)) |
| `models/flu-r1-2026-27{,-v2,-v3}.esl` | earlier versions, kept unchanged because experiments name them and running workers re-render them |
| `models/vscode-esl/` | VS Code syntax highlighting for ESL |
| `py-modules/flu_smh/` | `experiment.py` (experiment JSON, paths, iteration registry), `schedule.py` (vaccination schedules), `sim.py` (build, run, extract), `calib.py` (priors, draws, NHSN likelihood, scoring), `sw.py` (slurm-workflows objective), `proj.py` (members, restart mode, screen, early selection, guards, top-ups), `upscale.py` (severity and death replicates, all-deaths scaling) |
| `scripts/calib/` | `sw_calib.py` (multi-location explore and optimize on pilot jobs), `seed_from.py`, `flu_calib.py` (the first single-node driver) |
| `scripts/forecast/` | in-season forecasting: `fc.py` (snapshot, forecast, backtest, score, prescreen, weekly), `make_fc_experiments.py`, `pool.sbatch`; the study is flu-forecasting-FluTURE-2026-27 (`docs/forecasting.md` there) |
| `scripts/proj/` | `flu_proj.py` (members, run, select), `state_pipeline.py` (per-location chain from prepare to analysis), `make_seedx.py`, `make_prodv3.py` (v3, v3g, v4), `v5_pipeline.py` (top-up), sbatch wrappers |
| `experiments/` | one JSON per experiment (`<family>_<ST>.json`) and `iterations.json`, the registry of experiment families (it01-it12) used for report tags |
| `docs/flu.md` | model and calibration notes, iteration by iteration, through the round 1 submission |
| `docs/vaccine.md` | the vaccine mechanism, realized vs input VE, and how to measure a new parameter set |
| `legacy/` | COVID Round 20 and SPOK-24 models, drivers and docs this was built from (reference only) |

## Environment

```bash
module load gcc/14.2.0                      # in the same shell as the build, or libhdf5_cpp fails to link
eval "$($HOME/conda/bin/conda shell.bash hook)"; conda activate escape-sw   # escape-abm + slurm-workflows 5.1 + optimize
cd escape-flu-smh                           # experiment globs ($E=...) resolve against experiments/ here
```

## Commands

Most commands take experiment names or globs. The full chain is in AGENT_HANDOFF.md §5.

```bash
python scripts/calib/sw_calib.py submit explore <exps> ...             # calibration on pilot jobs
python scripts/proj/state_pipeline.py submit run <exps> --tag <tag> ...  # per-location production chain
python scripts/proj/flu_proj.py status <exps> --mode restart
python ../flu-round1-2026-27/notebooks/run_analyses.py <exps> --tag <tag> --upscale-to 300 --all-deaths
```

Every SLURM step prints or writes an sbatch command. Queue limits: the partition `bii` caps a user at 400 CPUs,
so wide runs use `--qos bii-half` or the partition `bii-largemem`.

## Lineage

[NSSAC/escape-covid19-smh](https://github.com/NSSAC/escape-covid19-smh) (COVID Round 20) →
[NSSAC/escape-cryptic-smh](https://github.com/NSSAC/escape-cryptic-smh) (cryptic round, SPOK-24) → this repo. This
repo was copied without history and started its own git history in Nov 2026.
