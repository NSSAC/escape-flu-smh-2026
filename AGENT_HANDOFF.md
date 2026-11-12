# Agent handoff: the ESCAPE flu model and pipeline

Context for an agent continuing this work or starting a new study with it: a later SMH round, an in-season
forecast, a vaccine-policy question or a retrospective. The first study was the Scenario Modeling Hub's Flu Round
1 2026-27, so the history and examples below come from it.

Read [README.md](README.md) and the pipeline's `README.md` and `docs/studies.md` first for the layout and the
study concept. This file adds what the code does not say: the history, why decisions went the way they did,
pitfalls, the owner's working rules, and open items. [docs/flu.md](docs/flu.md) is the long notebook through
prod-60-v2 (2026-11-05). [docs/vaccine.md](docs/vaccine.md) covers the vaccine retune. Written 2026-11-12.

## 1. Where things are

| what | where |
|---|---|
| this repo (model + drivers) | `/project/bii_nssac/biocomplexity/SMH/escape-flu-smh` (on GPFS: `/sfs/gpfs/tardis/project/...`) |
| this repo on GitHub | `git@github.com:NSSAC/escape-flu-smh-2026.git`, branch `main` |
| pipeline (data, analysis code, SMH export, all outputs) | `/project/bii_nssac/biocomplexity/SMH/flu-round1-2026-27`, its own git repo from 2026-11-12. Outputs are not tracked. Docs: `README.md`, `docs/studies.md`, `docs/smh.md`, `docs/initial-conditions.md` |
| hub clone the fits read | `flu-round1-2026-27/external/flu-scenario-modeling-hub`. **Do not pull it**: the fits must see the data as of the cutoff. Use `git fetch` + `git show origin/main:<path>` for newer files |
| hub resources | `external/flu-scenario-modeling-hub_resources` (`Rd7_datasets/`: coverage curves, CDC cumulative deaths) |
| hub PR checkout | `flu-round1-2026-27/submission/hub-pr` (sparse, blobless; managed by `submission/prepare_hub_pr.sh`) |
| scratch | `/scratch/$USER/flu-round1-2026-27/`: `calib/` (builds, input.h5), `sw/` (pilot-job scripts and logs, `sw/drivers/`), `ve/` (VE re-simulations), `detailed/` (sbatch + logs). **Scratch is purged**: nothing there is the only copy of a result, but the builds would have to be redone |
| Python | `module load gcc/14.2.0; eval "$($HOME/conda/bin/conda shell.bash hook)"; conda activate escape-sw` |
| R (hub validation) | `$HUBVAL_ENV`, default `/scratch/$USER/envs/hubval` (hubValidations + SMHvalidation) for `submission/validate_hub.R`. On scratch, so it can be purged: rebuild with `submission/envs/hubval_env.sh` |
| simulator | `escape_abm` (OpenMP) in `escape-sw`. A run is ~1.5 min per million people on a 40-core node. Pilot peak RSS ~63 GB (VA) |

## 2. Working with the owner

- **The owner runs every SLURM submission** (`sbatch`, `scancel`, `scontrol`). Write the sbatch script or command
  and hand it over. Do not submit.
- Commit or push only when asked.
- **Hub rule: no data after the fitting cutoff (2026-10-03 for this round) for fitting or selection.** Newer
  data is for validation only, such as `flu_analysis.refresh_truth()`, which reads CDC NHSN directly.
- Run anything heavier than a quick check through SLURM or on the owner's interactive node, not the login node.
- The owner likes paired comparisons (A/B/C vs D on the same member), figures named and embedded in self-contained
  HTML (the code-tunnel viewer does not load relative image links), and plain-language summaries with numbers.
- Before a long run, give a cost estimate in node-hours and wall time.

## 3. History

### Calibration iterations (`experiments/iterations.json`; reports in `flu-round1-2026-27/calib_reports/`)

| id | experiments | what changed / what was learned |
|---|---|---|
| it01 generic | `bt2023_*` | first backtest (fit 2023-24, project 2024-25). 5-13x too big, peaks 6-8 wk early, S0 0.96 |
| it02 informed | `bt2023i_*` | informed priors from the subtype-season analysis; long-term immunity (S0 0.62-0.80). Lost the spring tail |
| it03 lags-bg | `bt2023j_*` | seasonal lags, off-season background, sampled waning, ensemble WIS. Fit better, projection worse |
| it04 reduced | `bt2023k_*`, `bt2024k_*` | 7 searched dims + next-season draws; two backtests |
| it05 twostage | `bt2023l_*`, `bt2024l_*` | round 1 = fit season only; round 2 = Sobol over next-season parameters from restart |
| it06 subtype-weekly | `bt2023m_*`, `bt2024m_*` | + weekly per-subtype NB (w 0.3): subtype timing much better. Adopted |
| it07 production | `prod2025n_*` | 2025-26 fit -> 2026-27 scenarios; early-season screen and selection |
| it08 age-anchor | `prod2025o_*` | `hosp_age_mult_<band>` = NHSN / model age share (the model put 30% of admissions in 5-17 vs NHSN 5%) |
| it09 early-rise | `seedx*`, `slopex_FL` | H1N1 seeding pulse (model v3) for the September-rise states |
| it10 v4-vaccine | `prodv3_*`, `prodv3g_*` | hub-matched vaccine (model v4), rescaled rather than recalibrated; v3g re-guarded |
| it11 v4-recalib | `prodv4_*` | full recalibration under v4 (4 states). Too damped; not used |
| it12 v5-topup | `prodv5_*` | v3g topped up to >= 45 runs/state. **The current submission** |

### Production versions (analysis tags in `flu-round1-2026-27/analysis/<tag>/`)

| tag | experiments | status |
|---|---|---|
| prod-60-v2 | prod2025o_* + seedx2/seedx3 for 8 states | **round 1 submission, hub PR #57 (merged)**; archived in `submission/archive/2026-11-05-prod-60-v2/` |
| prod-v3 | `prodv3_*` | v4 vaccine on prod-60-v2's 60 members, tau x1.16, hosp x1.08; not re-guarded (15-35% peak before Nov) |
| prod-v3g | `prodv3g_*` | v3 + size and peak guard re-applied (29-59 runs/state); branch `UVA-escape_flu-2026-10-04-v3`, built, not submitted |
| prod-v4 | `prodv4_*` | recalibrated (VA MA WA FL); rejected |
| **prod-v5** | `prodv5_*` | v3g + top-up, 45-59 runs/state (median 48), 300 trajectories each via `upscale`; **update submitted 2026-11-09** on branch `UVA-escape_flu-2026-10-04-v5`; build copy in `submission/builds/prodv5/` |

Comparison of v2/v3/v3g/v5 (`notebooks/version_compare.py` -> `analysis/prod-v5/compare-versions/`, truth through
Oct 31):
- US reduction in admissions, B vs D: v2 20%, v3 37%, v3g 39%, v5 39%.
- State WIS sum: v2 2267, v3 4812, v3g 2899, v5 2839.
- v2 had the best WIS in 36 of 50 states, because its early weeks were selected under its own dynamics.

## 4. Key design decisions (and why)

- **Restart mode** (decided 2026-10-27, for cost). Each projection is one season from 2026-08-09 on the production
  person file. Per subtype, `ltimm_frac` is re-solved so that S0 on the origin (2026-10-04) equals that of the
  member's own calibration run. It is 0.35x the cost of re-simulating 2025-26, and the four scenarios of a member
  share the same history.
- **The simulator has no RNG seed** (`std::random_device` per thread). Identical configs differ from run to run,
  so paired A-D contrasts contain noise. Production deletes its h5 files, so any per-agent analysis needs new runs,
  kept on purpose (`notebooks/ve_runs.py`, `notebooks/detailed_analyses.py`).
- **Selection** goes screen (restart cut at origin + 7 d), then early-week NB score (tempered weights, ESS target),
  then full B, then the guard, then A/C/D for the kept runs.
  - **Guard:** weight 1 up to 1.5x the state's record season (total and peak, NHSN 2022-26), cos² down to 0 at 2x,
    drawn by systematic PPS.
  - **Peak guard:** no peak before 2026-11-01. No state peaked before November in NHSN 2022-26.
- **Vaccine (v4).** Under v2, realized VE was ~10% (infection), ~10% (symptomatic) and ~33% (hospitalization),
  against the hub's 50%. v4 sets `vax_escape_per_day` 1 (no escape), `vax_infection_ve` 0.43, `vax_symp_ve` 0.46
  and `vax_hosp_ve` 0, which gives ~23% / ~49% / ~52% (matched cohort, `ve_analysis.py`). The stronger vaccine
  removes 2025-26 transmission, so v3 rescaled it: tau x1.16 and hosp x1.08 (VA sweep).
- **Deaths.** The hub's `inc death` is CDC's all-deaths multiplier model, ~4.9x the model's in-hospital deaths.
  Submitted deaths are post-processed: admissions × an age profile of all deaths per admission
  (`notebooks/all_deaths_ratio.py`, centre 0.142, log-sd 0.20). See `upscale.Params.all_deaths`.
- **Severity uncertainty** comes only from post-processing (`flu_smh/upscale.py`: k replicates per run with
  national, state and AR(1) admission noise). `stochastic_run` = replicate + 1, and `run_grouping` = the run.
- **US = index-pairing sum of states.** Because the state ensembles are skewed, the US median comes out ~1.6x
  observed for every version, while the sum of state medians is ~1.0-1.1x. This is a known artifact, not a bug.

## 5. Running a study end to end (as Round 1 2026-27 ran it)

Run from `escape-flu-smh/` unless a step says otherwise. Notebooks live in `../flu-round1-2026-27/notebooks/`.

1. **Data.** `flu_data.py` (calendar: `ANCHOR`, `CALIB_END`; coverage `COVERAGE`), `surveillance_data.py`
   (NSSP, FluView, burden), `burden_infections.py` (infection histories by subtype), `subtype_seasons.py`
   (priors), `early_season.py` (evidence at the cutoff), `age_anchor.py`, `all_deaths_ratio.py`.
2. **Person files.** `run_person_parquets.py --anchor <tick-0 date> --vax-seasons ... --inf-seasons ... --tag
   <tag>` for the calibration anchor. Without `--tag`, it writes the production `data/person/<ST>-person.parquet`
   at the restart anchor. Batch: `run_initial_conditions.sh --slurm N`.
3. **Experiments.** `prod_experiments.py` (base) then `informed_priors.py prod2025_* --suffix <x> --twostage
   --early-season --subtype-weekly 0.3 --age-anchor --runs bt2023_VA bt2023_WA bt2023_MA`. **Add a row to
   `experiments/iterations.json`** for every new family.
4. **Per-state chain.** `scripts/proj/state_pipeline.py submit run <exps> --tag <tag> --peak-after <date>
   --keep-n 60 --min-keep 45 --max-topups 3 --upscale-to 300 ...` runs prepare -> explore -> members -> screen ->
   early -> B -> guard -> ACD -> analysis per state. It resumes and skips steps whose outputs exist. Status is in
   `analysis/<tag>/pipeline_status.csv`.
5. **Analyses.** `run_analyses.py <exps> --tag <tag> --upscale-to 300 --all-deaths [--states US --us always]`
   writes `analysis/<tag>/` (fig01-fig13, per-figure pages, index.html). Use `version_compare.py` to compare
   versions.
6. **Hub file + PR.**
   - Use `submission/v5-update/build_v5_update.sh` as the template. It checks completeness, runs the US analysis,
     then `smh_submission.py`, then the abstract (`abstract_results.py <tag> --write` + `fill_tokens.py`), then
     `prepare_hub_pr.sh --commit`.
   - Nothing is pushed. The owner pushes the branch and pastes `submission/PR_DESCRIPTION.md` into GitHub.
   - Validation passing "except `flat_projection`" is expected: small states have all-zero age-group runs.
7. **Per-agent ("detailed") analyses.** `notebooks/detailed_analyses.py {pick,extend,show,sbatch,run,analyse}`.
   - A selection (`analysis/detailed/selections/<name>.json`) fixes members × replicates.
   - Runs are kept permanently in `flu-round1-2026-27/proj_detailed/`.
   - Analyses: `generations` (5×5 generations-to-outcome matrices, by ancestor vaccination) and `paired` (the same
     person across scenarios: total, indirect and direct effects by generation).
   - `VA-core10` = prodv5_VA s0023/s0064/s0037 × 10 replicates × A-D. Direct effect of one's own dose on causing
     a hospitalization: about -24% at every generation.

## 6. Pitfalls (each of these cost time)

- **`module load gcc/14.2.0` in the same shell as the build**, not inside a pipe or subshell, or libhdf5_cpp fails
  to link (CXXABI).
- **The simulator silently ignores unknown config names.** A typo is a no-op. `ve_runs.py plan` checks names
  against the ESL; do the same for any new parameter.
- **Never edit a model ESL that runs are using.** Workers re-render the model into their build dirs on every run.
  Make a new version file (`-v5.esl`) and point the experiments' `model` at it.
- **Build `$E` (the experiment list) inside `escape-flu-smh/`.** Globs resolve against `experiments/`, so building
  it elsewhere gives "no experiments".
- **SLURM limits.** Partition `bii`'s default QOS caps a user at 400 CPUs: use `--qos bii-half` (2200) or
  `bii-largemem`, which has no per-user cap. Production reservations can pause pilots.
- **Hung drivers.** Pilots can die while ds-service still lists tasks as "running" forever. Check with
  `swtop --plain <addr>` (tasks running vs. the pilots' last log lines). To recover: hold the dependent job, cancel
  the driver and its pilots, resubmit (`flu_proj` resumes), and re-point the dependency.
- **Do not submit an array twice.** `detailed_analyses.py` now locks each result (`.lock`, per-job partial h5,
  verify, then promote). Older drivers do not.
- **Data gotchas.**
  - `load_nhsn` includes a US row, so summing all locations double counts.
  - The hub's target data drops rows for up to 18 states a week, so validate against CDC directly
    (`refresh_truth()`).
  - KS and AR stopped reporting in mid-2026; selection falls back to equal weights for them.
  - FluView labels a season with a trailing space.
  - `fa.truth_missing("US")` is wrong: test per state.
- **Averaging indicators.** For "P(at least one)" over replicates, average the per-replicate indicators. The
  indicator of averaged counts means "in ANY replicate" and grows with the number of replicates (fixed in
  `detailed_analyses.paired` on 2026-11-10).
- **`pgrep -f` / `pkill -f` match your own shell's command line.** Use the `[d]etailed` bracket trick.
- **`min_improvement` in sw optimize** is a fraction of |log(-score)|, so 1% is ~27 score points. Use
  `--min-rounds 6 --patience 4 --min-improvement 0.002`.

## 7. Open items and loose threads

**Repository / housekeeping**
- Done 2026-11-12:
  - the pipeline got its own repo and docs;
  - Round 20 leftovers moved to `legacy/`;
  - `docs/vaccine.md` written;
  - `/scratch/bl4zc` paths replaced with `$USER` / `%u`;
  - the hub-validation env rebuild recorded in `submission/envs/hubval_env.sh`, which has not been re-run.
- **Study constants are still scattered.** Dates, scenario letters and the D reference sit in several files
  (pipeline `docs/studies.md`, "Still hard-coded"). Reading them from the experiment's `production` block would
  make a new study a config change.
- **A study's outputs live inside a checkout of the pipeline repo** (`FLU_ROUND`). Separating code from study
  directories would let one checkout serve many studies.
- **The analyses assume scenarios A-D with D = no vaccination.** Generalize them to any scenario set with a
  named reference.
- **The fitting data come from the hub clone's target data.** A hub-independent NHSN snapshot loader (CDC
  `ua7e-t2fy`, cut at `data_end`, saved with its retrieval date) is needed for studies outside SMH.
- `docs/flu.md` "Still to do" is an October snapshot, superseded by this list.

**Model / method (for the next round)**
1. **Recalibrate under the v4 vaccine instead of rescaling.** v3 is a uniform tau/hosp rescale: it fixes size,
   not shape, and moved early growth. The prod-v4 recalibration looked too damped. This is the main model debt.
2. **The 2025-26 vaccination history uses stand-in coverage** (Rd6 sc_A, Rd5 sc_C_D). A real 2025-26 source is
   still needed.
3. **Early-season dynamics.** The September rise followed by an October slowdown (FL, LA, MT, HI) is not
   reproduced. The Nov-1 peak guard is a post-hoc fix. Candidates: a next-season seasonal amplitude, or seeding
   that can make a slow early rise.
4. **Age structure.** Without `hosp_age_mult`, 5-17 is overrepresented (53% of infections). The multipliers fix
   admissions, not transmission. Subtype-specific multipliers could help (50-64 overshoots in H3 seasons).
5. **The objective.** `fit_subtype_ll` contains a constant entropy (~185-210) that compresses log(-score).
   Subtracting it was proposed but not done.
6. **US aggregation.** Index-pairing sums inflate the US median (§4). Consider rank pairing, or report the
   difference.
7. **Severity** is fixed in the simulation (hosp_prob_scale, age multipliers). Its uncertainty is only the
   post-processing replicates.
8. **Seeds.** Pairs would be far less noisy with a simulator seed option. Ask the escape-abm developers.

**Scoring the submitted projections as the season unfolds**
- `version_compare.py` with refreshed truth (`--refresh-truth` in `run_analyses.py`) re-scores v2/v3/v3g/v5.
- The detailed-analysis selection can be extended to other states:
  `detailed_analyses.py pick <name> prodv5_<ST> --quantiles 0.25 0.5 0.75 --replicates 10`.

## 8. Starting a new study (a later SMH round or anything else)

The full checklist is in the pipeline's `docs/studies.md`. In short:

1. **Directory.** Make a new checkout of the pipeline repo for the study's outputs, and `export FLU_ROUND=<it>`.
   Fetch `external/` if the study uses hub data or coverage curves.
2. **Calendar, campaigns, scenarios.**
   - In `flu_data.py`: `ANCHOR`, `CALIB_END`, `PROJ_START/END`, `COVERAGE`.
   - In each experiment: the `production` block (`origin`, `data_end`, `scenarios`, `restart_anchor`).
   - Then the remaining hard-coded constants listed in `docs/studies.md`.
   - For an SMH round, read the hub README first. This round it superseded the PDF (dates and coverage seasons
     changed).
3. **Model.** Start from v4 in a new version file (`flu-r<k>-...esl`), and keep the old ones unchanged.
4. **Inputs.** Person files at the new anchors, then the experiments (new family names), then the
   `iterations.json` rows.
5. **Backtest the changes first.** Use at least two seasons (as it04-it06) on 3 states (VA, WA, MA) before
   running all 50.
6. **Outputs.** The native outputs and analyses need no export. For SMH, follow the pipeline's `docs/smh.md`: copy
   `submission/v5-update/` to a new drafts folder, then run the build script with `PREFIX`, `TAG`, `DRAFTS` and
   `BRANCH`.
