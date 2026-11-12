# SMH Flu Round 1 2026-27

> This notebook runs through prod-60-v2, the round 1 submission (2026-11-05). For later work (the v4 vaccine, prod-v3/v3g/v5, the detailed per-agent analyses), the current status and the open items, see [AGENT_HANDOFF.md](../AGENT_HANDOFF.md).

* Hub: https://github.com/midas-network/flu-scenario-modeling-hub (round set 2026-10-01; README is authoritative)
* Resources: https://github.com/midas-network/flu-scenario-modeling-hub_resources (`Rd7_datasets/`)
* Our data and analysis workspace: `/project/bii_nssac/biocomplexity/SMH/flu-round1-2026-27`. See its README
  and `notebooks/flu_initial_conditions.ipynb`.

## The round (hub README, 2026-10-01; supersedes the Sep 17 PDF)

| | |
|---|---|
| fitting data | ends between Sat 2026-09-26 and Sat **2026-10-03** |
| projection | Sun **2026-10-04** → Sat 2027-06-05, 35 weeks (`origin_date` 2026-10-04) |
| due | Tue 2026-10-20 |
| VE | 50% against medically attended illness and hospitalization, all ages; VE vs infection is ours (lower) |
| A `A-2026-09-29` optVax | 2019-20 coverage |
| B `B-2026-09-29` usualVax | **2024-25** coverage |
| C `C-2026-09-29` usualVaxAdults_noVaxChildren | 2024-25 coverage for 18+, none for 6mo–17y |
| D `D-2026-09-29` noVax | none (no curve provided) |
| required targets | `inc hosp` samples (300–600 trajectories, paired A–D) at 0-130 (age groups optional), and **S0** per trajectory on 2026-10-04 |
| optional | `inc death` (national, CDC multiplier model), `inc ed visit`, quantiles, peak size and time |

Coverage curves: `Rd7_datasets/flu_RD7_Vaccination_curves.csv` (geography, season_survey_year, age_group 0.5-4 /
5-12 / 13-17 / 0.5-17 / 18-49 / 50-64 / 65+, pop, date, scenario A/B/C, coverage in percent). They **start on
2026-08-15**, seven weeks before the projection, so scenario vaccination is already under way when the projection
starts. The hub says to calibrate to scenario B if the early data conflicts with a scenario, and every trajectory
in a matched set must share its initial calibration conditions.

The goals are the impact of vaccination in high- versus low-burden seasons (split on scenario D's epidemic size) and
the direct and indirect benefit of child vaccination (B vs C). That makes **transmission from children, and indirect
protection, the core model behavior**, not a detail.

## The model: `models/flu-r1-2026-27.esl`

A first flu version of `covid19-r20.esl`, with three subtype contagions and subtype-specific severity. It parses,
builds and runs on VA against the subtype-resolved person file. A 120-tick run reproduces S0 from the person file
to five decimals, and per-subtype admissions, P(hosp | symptomatic) and P(death | hospitalized) as configured.

* **Timeline.** Tick 0 = **Sun 2026-08-09**, the week before the coverage curves start, so all scenario vaccination
  happens in-sim with its real timing. Calibration runs through tick 55 (Oct 3), the projection is ticks 56–300, and
  S0 is read at tick 56. The person file must be built with this tick 0.
* **Three contagions**, one per subtype: `flu_A_H1N1`, `flu_A_H3N2`, `flu_B` (underscores: hyphens do not
  parse). They share the state enum (S V E I A H D R), the vaccine (one trivalent shot), severity, natural
  history, seasonality and tau. Each subtype has its own infection clocks, admission flag, `tau_scale_<sub>`,
  `seed_share_<sub>` and statistics. Infection protection is homologous (`infection_ve` 0.70) plus a **cross-protection matrix**
  (`xprot_<from>_to_<to>`): H1N1 and H3N2 protect each other by 0.20, A and B not at all, with the same waning and
  drift as homologous protection. Each
  subtype's clock `last_infection_tick_<sub>` is a static field loaded from the person file (2022-10 → tick 0
  history from flu-round1-2026-27 `burden_infections`, below). Death in any subtype removes the agent from
  the others, and co-infection is allowed.
* **Immunity.** Vaccine and infection protection are separate, multiplicative terms. Vaccine: VE 0.25 vs
  infection and 0.3333 extra vs hospitalization (50% total), waning half-life 150 d, escape from the
  formulation tick. Infection: 0.70 vs reinfection after R, half-life 365 d, escape from the infection tick.
  Drift is 30%/yr. R lasts ~180 d (the rest of the season).
* **Severity, by subtype and age** (flu-round1-2026-27 `subtype_severity.py` + `burden_infections.py` →
  `data/processed/infections/model_esl_params.csv`; notebook `burden_infections.ipynb` §4–5, §8):
  * P(hosp | infection) = A(H1N1) ladder × `hosp_rel_<sub>_<age>`, ÷ `symptomatic_frac` at I → H. The ladder
    is NHSN admissions per infection: 0-4 0.58%, 5-17 0.36%, 18-49 0.46%, 50-64 0.82%, 65+ 5.8%.
  * The relative IHRs are **within-age**, from FluSurv-NET 2010-19 (Sumner et al., *Lancet Microbe* 2023) with
    infections from CDC's burden-model inputs × NREVSS, age-tilted by FluView public-health-lab ages. H3N2 is
    0.76 / 0.84 / 0.89 / 0.93 × H1N1 (0-17 / 18-49 / 50-64 / 65+), and B is 0.69 / 0.75 / 0.80 / 0.95 ×.
  * H3N2's ~1.4× all-age excess is age composition (it infects older people), which the ladder and the
    person file's subtype-by-age history already carry. Do not add an all-age multiplier on top.
  * P(death | hospitalized) = `death_hosp_<sub>_<age>` × `death_prob_scale`: FluSurv-NET in-hospital deaths
    per admission by age, split with Sumner's age-specific ORs vs H3N2. H1N1 is deadliest in hospital
    (18-49: 2.0% vs H3N2 1.1%), and B in children (0.96% vs 0.38%).
  * The same IHRs back-calculate the person file's infections (NHSN admissions ÷ IHR), so the model and its
    initial condition are consistent by construction.
* **Vaccination.** Scenario schedule rows `tick,new_vax_count,min_age,max_age,vax_campaign,vax_formulation_tick,
  propensity_max`. S/E/A/R are eligible and only S moves to V. `propensity_max` puts the campaign on habitual
  vaccinators (low `vax_propensity`). The column is now spelled `vax_formulation_tick` (Round 20 had `forumlation`).
* **Outputs.** `new_hosp_count_<sub>` counts each hospital entry once (Round 20's version was a lagged census),
  and `new_hosp_count` is the all-subtype sum (the hub target). Admissions are reported one tick late.
  Also `new_hosp_with_recent_vax_count`, and `susceptible_mass_<sub>` (per-subtype S0 × N, for the hub's
  optional S0_AH1 / S0_AH3 / S0_B; the required S0 weights them by each subtype's share of cumulative
  infections). Per-subtype state counts, transitions and interventions are under each contagion's HDF5 group.
* **Removed** from Round 20: winter double-peak scaling, the death-prob parquet, the high-risk split, and the
  mid-trajectory initial state with its burn-in and infectivity ramp.

## Calibration and backtesting: `py-modules/flu_smh`, `scripts/calib/flu_calib.py`

One **experiment** = `experiments/<name>.json` (state, calendar, person file, in-sim campaigns, priors,
likelihood). One **particle** = one continuous run from the experiment's tick 0. The first two experiments are a
backtest: fit 2023-24, project 2024-25 out of sample (`bt2023_WY`, a small test; `bt2023_VA`).

```bash
module load gcc/14.2.0 && conda activate escape
python scripts/calib/flu_calib.py prepare bt2023_VA    # vaccination schedule, build, input.h5 (once)
python scripts/calib/flu_calib.py sample  bt2023_VA    # scrambled Sobol over the priors -> particles.csv
python scripts/calib/flu_calib.py submit  bt2023_VA --nshards 64      # SLURM array (flu_calib.sbatch)
python scripts/calib/flu_calib.py score   bt2023_VA    # NB likelihood + subtype mix -> scores.csv
```

* **Model additions for calibration:** seasonality is a config-driven cosine (`seas_peak_tick`, `seas_power`),
  so no rebuild is needed per particle. A season block (`season_break_tick`, `tau_scale_<sub>_next`,
  `seed_share_<sub>_next`) lets one run fit a season and project the next with a different subtype mix.
* **Inputs:** person files with any anchor (`run_person_parquets.py --anchor 2023-08-06 --vax-seasons 2021-22
  2022-23 --tag bt2023`). In-sim campaigns come from `flu_data.COVERAGE` (the real 2021-22 .. 2024-25 campaigns,
  each a hub "same as season X" file shifted 728 days onto X's dates).
* **Outputs:** `flu-round1-2026-27/calib_<name>/` holds `particles.csv`, `runs/p####.parquet` (weekly
  infections, admissions and deaths by subtype × age), `p####.stats.parquet` (incl. S0), and `scores.csv`. The build,
  input.h5 and raw HDF5 are on `/scratch/$USER/flu-round1-2026-27/calib/<name>/`, and the HDF5 is deleted after
  extraction. The analysis notebook is `flu-round1-2026-27/notebooks/calibration_backtest.ipynb`.
* **Cost:** WY ~3–6 min per 693-tick run. VA ~14× the agents, so roughly 1–2 h per run on 40 cores. 128 VA
  particles over 64 array tasks is about 2–4 h of wall time.

### Informed priors and model v2 (`bt2023i_*`)

`flu-round1-2026-27/notebooks/informed_priors.py` turns the subtype-season analysis
(`subtype_seasons.ipynb`: FluView 2010-26, true season = at least 10% of a season's positives) into
state-specific experiments: `python informed_priors.py bt2023_VA bt2023_WA bt2023_MA` writes
`experiments/bt2023i_<ST>.json`. Each prior's source is in the JSON's `provenance`; the table is in
`flu-round1-2026-27/data/processed/informed_priors/`.

* **Model v2** (`models/flu-r1-2026-27-v2.esl`, chosen by the experiment's `model` field) gives each subtype its
  own seasonal phase: H1N1 peaks at `seas_peak_tick`, H3N2 and B `seas_lag_<sub>` days later. The season
  block also moves the peak (`seas_peak_tick_next`, an absolute tick) and the lags (`seas_lag_<sub>_next`).
  With its defaults it reproduces v1 (checked on WY: v1-v2 differences equal run-to-run noise). v1 stays
  in place while runs that use it are in flight; promote v2 once they finish.
* **Information cutoff.** Fit-season priors may use the fit season's own data, which the likelihood also
  sees. Next-season priors use only seasons up to the fit season, so a backtest projection stays out of
  sample.
* **Timing.**
  * `seas_peak_tick` comes from the state's NHSN peak minus the forcing-to-admissions offset measured in
    the runs (about 30 days).
  * `seas_peak_tick_next` comes from the state's historical lab peaks moved onto the next season.
  * The lags come from the fit season's observed H3 - H1 and B - H1 gaps; the `_next` lags from their
    historical spread.
* **Subtype mix.** `tau_scale_*` are no longer sampled directly. The search samples target weights
  `share_w_<sub>` (fit season: around the likelihood's observed mix) and `share_w_<sub>_next` (a mixture:
  "absent" with probability 1 - p, "true season" with p = P(true season | the state's last three-season
  pattern)). `share_map` converts them to `tau_scale_*`: a fitted elasticity of about 6.4 for
  log(share ratio) on log(tau ratio), R^2 about 0.75. The three are normalized to a geometric mean of 1.
* **Mixture priors** (`{"mix": [[lo, hi, w], ...], "scale": "log"}`) are sampled by inverse CDF. Sobol and
  BO search their unit coordinate, and the run JSON records the actual value.
* **Staggered seeding (v2).** `seed_start_<sub>` (+ `_next`) opens a `seed_days` window in which the subtype
  seeds at the full rate, with `seed_floor` × that rate outside it. The priors come from each state's
  observed or historical onsets (start of the main wave, about 7 weeks before the peak), 7-42 days earlier.
  In `bt2023i` the seasonal lags were off, on the reasoning that they move the same waves as the windows.
  That was wrong: see round 3 below.
* **S0 and long-term immunity (v2).**
  * S0 = `susceptible_mass_<sub>` / N is the population's mean relative susceptibility (1 = naive).
  * The person file's recent history alone gives about 0.95, even with no waning at all about 0.81
    (backtest) or 0.74 (production). The model keeps only each subtype's last infection, and each season
    infects 3-8% per subtype.
  * The hub cites 0.65-0.70 (Yang et al. 2015), which reflects lifetime exposure.
  * `ltimm_frac_<sub>` flags that fraction of adults (`ltimm_child_rel` = 0.5 of it for children) at tick 0
    with 1 - `ltimm_protection` (0.8) susceptibility. The informed experiments sample one `ltimm_frac`,
    solved per state for S0 0.62-0.80.
  * Waning is slowed to a 2-year infection half-life and 4-year drift half-life (a combined 240 days
    before). Round 3 samples the infection half-life instead.
  * Transmissibility priors scale by S0_history / S0_target (about 1.33), and `hosp_prob_scale` is held
    to 0.8-1.25.
* **Reports** show attack rate and S0 per run against the hub ranges, and plot lack of fit on a log axis.
  On a linear axis, a few very bad runs flattened all the good ones into one line, which looked like
  "many values fit equally well".

### Round 3 (`bt2023j_*`): lags, background, sampled waning, ensemble skill

`python informed_priors.py bt2023_VA bt2023_WA bt2023_MA --suffix j`. The `bt2023i` review found two
failures. The model lost the spring tail: every subtype peaked in January, while NHSN stayed at 110-240 a
week through April. The 2024-25 projection came out too small and too late (top-20 median 0.23-0.63× the
observed total). Four changes:

* **Seasonal lags are back on by default** (`--no-lags` drops them). The seeding windows set when a subtype
  is introduced, but only a later forcing peak keeps H3N2 and B going into spring. The fit-season priors are
  the observed gaps ± 14 d: VA B +10 wk; MA H3N2 and B +8 wk; WA H3N2 +5 wk, B −2 wk.
* **Background admissions in the likelihood.** `likelihood.background = {"fit": b, "proj": b}` is added
  to the model's weekly mean (`calib.background`). It covers sporadic off-season admissions that the model,
  whose transmission dies out between seasons, does not simulate. b is the median NHSN week of the latest
  summer (May 15 - Jul 31) with at least 4 reported weeks before the window's start: VA 30, WA 18 / 20,
  MA 9. NHSN's May-Oct 2024 gap sends VA and MA back to 2023 for the projection. Scores with a background
  are not comparable to `bt2023i` scores.
* **Infection waning is sampled**, as `infection_half_life_days` [182, 1095] (log; `calib.configs_for`
  converts it to `infection_waning_per_day`). Drift stays fixed at 4 y. History-only S0 moves only
  0.96 → 0.95 across the range, so the fit season barely constrains waning. The projection's carryover
  does, so judge it on projection skill. `ltimm_frac` is solved to keep S0 in 0.62-0.80 at both ends of
  the range.
* **Ensemble projection skill.** `calib.ensemble_skill` scores the top N runs by fit score (default 20) as
  one equally weighted ensemble, as the hub scores a submission. It reports WIS over the 23 hub quantiles,
  compared with the best run's MAE (its one-member WIS), 50/90% coverage, the season total and peak against
  the members' spread, and the members' subtype mix and "absent" draws. The reports add `ensemble.png` and
  `ensemble.csv`, and the summary page adds a table (`calib_report.py --ens-top N`). On `bt2023i` the top-20
  projection WIS was 104 (VA), 71 (WA) and 225 (MA).
* Each round gets its own Sobol seed (the state code plus the suffix; `i` keeps the `bt2023i` seeds), and
  27 dimensions per state.

### Round 4 (`bt2023k_*`, `bt2024k_*`): two backtests, a reduced design, next-season draws

`bt2023j` fit 2023-24 well, but its 2024-25 projection was worse than `bt2023i`'s (top-20 WIS 164/113/247):
too small and 4-8 weeks late. Fit skill was unrelated to projection skill, and 21 of 27 parameters had no
detectable effect on the fit (|Spearman rho| < 0.2).

* **A second backtest:** `bt2024_<ST>` fits 2024-25 and projects 2025-26 (anchor 2024-08-04, person files
  `data/person/bt2024/`, `--vax-seasons 2022-23 2023-24`). The 2025-26 in-sim campaign is a stand-in
  (`flu_data.COVERAGE`: 2023-24 coverage on 2025-26 dates). 2025-26 was H3N2-dominated (about 70% of
  infections), peaked in early January in VA and MA, and was 0.44-0.65× the size of 2024-25.
* **Reduced design** (`informed_priors.py`, the default; `--full` restores round 3's):
  * 7 searched: `baseline_tau`, `tau_seasonality_scale`, `seas_peak_tick`, `seas_power`,
    `baseline_seed_prob`, `share_w_A_H3N2`, `share_w_B`.
  * 10 drawn per run (`exp.draws`; `calib.draws_for` seeds them from the experiment seed and the run id): every
    `*_next` parameter plus `tau_next_mult`, the projected season's intensity, which scales its three
    `tau_scale_*_next` (log [0.75, 1.75]). The fit cannot see these, so searching them only made the
    optimizer's job harder. Drawing them makes the top runs a prior-predictive ensemble for the next season.
  * Fixed: `hosp_prob_scale` 1. Fit-season lags at the observed gaps. Seeding windows at the middle of the
    onset window, `seed_days` 70, `seed_floor` 0.1. Infection half-life 365 d. `ltimm_frac` solved for a
    tick-0 S0 of 0.70. `share_w_A_H1N1` at the observed share (only ratios of the three weights matter).
* **Background as a floor:** `likelihood.background_mode = "floor"` (mean = max(model, b)). The additive
  form also counted in-season, about 30% of WA's 2023-24 total.
* **Next-season timing:** `seas_peak_tick_next` uses the forcing-to-peak offset measured in `bt2023j`'s
  projected seasons (38 d; VA 33, WA 38, MA 46) instead of the fit season's 29 d.
* **History-weighted ensemble** (`calib.intensity_weights`, `ensemble_skill(weights="history")`). Members are
  re-weighted so the model's next/fit season admissions ratio follows the CDC burden record's
  season-to-season hospitalization ratio (2010-25, n = 10: geometric mean 1.10, sd(log) 0.72, p10-p90
  0.47-1.95). This uses the national record, not the projected season, so both backtests stay out of
  sample. Reports show the equal-weight and weighted projections side by side, with the weights'
  effective sample size.
* Next-season information is pooled across the two backtests, three states and two projected seasons.
  That pooled information sets the production (2025-26 → 2026-27) draws.

### it05 (`bt2023l_*`, `bt2024l_*`): two-stage design

it04's lesson: next-season parameters cannot be seen by the fit, so searching or drawing them inside the
calibration only muddles both scores; and fit-shape parameters that were fixed (the fit-season lags) left fits out of
reach (bt2024k WA's two 2024-25 peaks). So a parameter is calibrated in round 1 only if it moves the fitted path.

* **Round 1 = the calibration experiment** (`informed_priors.py ... --suffix l --twostage`): it04's 7 searched +
  `seas_lag_A_H3N2` / `seas_lag_B` (observed gap +- 8 weeks), no draws, no season break, no projection window; runs end
  on the next season's origin (first Sunday of October: 2024-10-06, 2025-10-05; production 2026-10-04), ~0.6x a
  two-season run. Calibrate with `sw_calib.py explore|optimize bt2023l_* bt2024l_*` as before (fit score only).
* **Round 2 = `production["next"]`**, swept by `flu_proj.py members --design sobol --top 5 --per-fit 32` (the top 5
  fits x the same 32-point Sobol design, 160 members per experiment), `flu_proj.py run` (restart mode from the next
  season's person file, `production["restart_person_file"]`: bt2024 files for bt2023l, prod2025 for bt2024l),
  `flu_proj.py score` (`proj_<exp>/restart/B_scores.csv`) and `notebooks/next_season_report.py it05` ->
  `calib_reports/it05-twostage_next.html` (prior-predictive ensemble skill, within-fit effects, where the best 20%
  sit in each prior, fans).
* **Round-2 parameters:** it04's draws, wider (`tau_next_mult` 0.5-2, `seas_peak_tick_next` +- 28 d around historical
  lab peaks) + `escape_next_<sub>` 0-0.4: immunity lost at the season break (a drift jump, e.g. 2025-26's H3N2
  subclade K), applied in restart mode as S0 target 1 - (1 - S0_k)(1 - e_k) (proj.restart_ltimm).
* **Reading round 2:** structural quantities (the timing offset, intensity, carried-over immunity, escape) are the
  same every season: if their good region sits off-centre the same way in both backtests, move the production prior.
  Season-specific ones (timing, mix) stay draws; the backtests check that the observed season is inside the spread.
* **Production (it06, `prod2025l_*`):** the same round 1 for 2025-26 (runs to 2026-10-04); round 2 is the projection
  itself: `members --design random --top 20 --n 300` from the revised `production["next"]`, scenarios A-D.
* Inputs are shared: `bt20XXl_<ST>/input.h5` and `bt2023l_<ST>-restart-input/input.h5` link to the bt2023/bt2024k
  inputs on scratch (same person files and network).

### Production 2026-27 (it07): early-season evidence, screening, selection

The projected season started before the origin, and the hub lets us fit it up to Sat 2026-10-03 ("end date for
fitting data"; `fd.CALIB_END`, `production["data_end"]`). Nothing after that date is used (`nssp_state.csv`
already holds 2026-10-10: cut).

* **Evidence** (`flu-round1-2026-27/notebooks/early_season.py` -> `data/processed/early_season/evidence_2026-10-03
  .csv/.png`). National NHSN 3334 admissions/week on 2026-10-03 vs 921-1024 at that week in 2022-23, 2023-24,
  2025-26 (~3.3x), growing ~29%/week; H1N1 89% of subtyped A Aug 1 - Oct 3 (every HHS region 79-100%); clinical-lab
  positivity 3.65% vs 0.5-0.9%. 2022-23, the last early season, was at a quarter of this level on Oct 1 and peaked
  in early December. Uneven: WA 22x the same week, OR/MT/UT/HI 12-15x, CA 5x; VA 0.8x, MA 1.8x. H1N1 >= 83% of A
  in every state. State weeks after ~Sep 26 are incomplete (33 of 51 reported Oct 3).
* **Subtype shares, Aug 1 - week 39** (fixed 2026-10-28, `burden_infections._state_h3`): a state's weekly H1:H3 is
  its HHS region's odds x one constant per FluView season (fit to the state's season total). FluView seasons run
  week 40 .. 39, so Aug - Sep 2026 sat in FluView 2025-26 and took its H3-heavy constants (DC x16.7, RI x12, KS x54):
  DC's early H1N1 share of A 0.48 against its region's 0.96, CT 0.35, RI 0.31. Now the constant is solved on and
  applied to the weeks where the FluView and epidemic (Aug-Jul) seasons agree; the others keep the region's odds.
  Past fitted seasons move little (state-season H1N1 share, admission-weighted: median <= 0.009, max 0.075). Unsubtyped
  A (2-10% of A) was and is split in the week's subtyped H1:H3 ratio. **To do after the it06 optimize** (whose
  workers read these files): rerun `burden_infections.py` to rewrite `data/processed/infections/`;
  `early_season.regional_tail` applies the fix to the old file meanwhile (a no-op on the new one).
* **Assumptions** (state them in the abstract metadata): an H1N1-dominant season already under way, early where the
  data show it; H3N2 and B may follow but are minor (capped); intensity and H1N1 drift wider than history.
  Concretely, `informed_priors.py ... --twostage --early-season` replaces the history priors of
  `production["next"]` (`early_season.next_priors`): H1N1 seeding from the restart anchor 2026-08-09 (to +35 d
  where admissions are >= 1.5x the same week of earlier seasons, else to the history's upper end; the history
  prior started 2026-09-18, but the Aug 9 person file holds only ~0.04% recent infections, so every admission
  since August must come from next-season seeding); `share_w_A_H1N1_next` 0.45-0.85 with no "absent" segment
  where H1N1 >= 70% of A (the history gave WA's H1N1 a 66% chance of being absent); H3N2 "absent" weight 0.5, true
  share capped 0.35; B capped 0.3; `seas_peak_tick_next` 28 d earlier at its lower end; `tau_next_mult` 0.5-3;
  `escape_next_A_H1N1` 0-0.5.
* **Screen** (`proj.py` mode `screen` = restart cut a week after the origin, ~1/5 of a full run: MA 204-228 s vs ~1150 s): candidates =
  top round-1 fits x a Sobol design over `production["next"]`, scenario B only, scored on the early window
  (`proj.early_window`: restart anchor .. data_end) by `early_score` = all-age weekly NB + `early_subtype_weight`
  (default the calibration's `subtype_weekly`, else 0.3) x by-subtype weekly NB (NHSN x lab share).
* **Select** (`flu_proj.py select --mode screen --weights early --n 300 --ess E`): weights ~ exp(early_score / T),
  T set so the effective sample size is E (`proj.tempered_weights`; T = 1 would collapse onto a few members on 8
  noisy weeks); 300 systematic draws with replacement -> new members s0000.. in `restart/selected.csv` (a repeated
  parameter set is a new trajectory: the simulator has no seed). `<mode>/selection.csv` keeps every candidate's
  score, weight and copies.
* **Assess the rule on the backtests** (`next_season_report.py it05|it06`, "Selection on the early weeks"): each
  round-2 member's early_score (anchor .. the day before the origin) weights the ensemble at several ESS, scored
  on the weeks AFTER the origin only. The ESS with the best post-origin WIS (and 90% coverage still ~0.9) sets E.
  Only bt2024* test it: bt2023*'s early weeks (2024) fall in NHSN's voluntary-reporting gap.
  it06 (2026-10-29): ESS 25-50% of the candidates cut post-origin WIS by 30% (VA) to 60% (WA), 90% coverage ~0.8
  unchanged; below ~15% WA degrades. Caveat: those early weeks sat at the likelihood's background floor, so every
  quiet member scored the same and the rule only removed members that took off early. 2026-27's early weeks are
  well above background in most states, where the score ranks members continuously: keep E at ~20-25%.
* **Guard** (`flu_proj.py select --mode restart --weights guard --n 300`, after a full B pass of the screened
  members): weight 1 up to 1.5x the state's largest NHSN season since 2022-23 and cos^2 down to 0 at 2x it, the
  lower of the season total's and the peak's weights (a season missing weeks counts as a lower bound;
  `proj.guard_weights`, `GUARD_FULL`, `GUARD_MAX`); kept by systematic PPS sampling (`proj.pps_sample`: inclusion
  probability ~ weight) among weight > 0, so the screen selection oversamples (3x). Relaxed 2026-10-30 (was: full
  weight only to 1x, total x peak weights, Efraimidis-Spirakis sampling): preview-v3 VA kept 2 of the 20 members
  whose peak was 1-2x the record; the relaxed rule keeps ~7 of 30 above the record (90% season total 1.31 -> 1.54x). The record is 2024-25 in 48 of
  50 states; DC's and WY's record season misses winter weeks (warned: their guard may be tight).
  `restart/screened.csv` keeps the screen selection; `restart/selected.csv` becomes the kept members.
* **Priors, 2026-10-29.** The first test-state preview (`calib_reports/..._preview_v1.html`, runs in
  `proj_prod2025n_<ST>_v1`) projected 5-15x last season: tau_next_mult 0.5-3.0 and escape_next_A_H1N1 0-0.5
  put the selected p90 at ~2x last season's tau x S, and the early weeks (low counts) could not cut members that
  take off after the origin. Now tau_next_mult 0.6-1.6 and escape_next_* 0-0.3 (the it06 backtests' good region;
  `early_season.TAU_NEXT`, `ESCAPE_NEXT`); only production["next"] changed, so round-1 runs are unaffected.
* **Preview pipeline** (`scripts/proj/preview_pipeline.sbatch "<exps>" <tag> ["<exps through the guard>"]`): one driver job runs explore -> members -> screen -> early select -> B -> guard -> A C D -> run_analyses.py, each step resumable; header has the sbatch line and the env knobs.
* **Per-state pipeline** (`scripts/proj/state_pipeline.py submit run prod2025o_* --tag <tag> --jobs 54 --qos bii-half`): the same steps as the preview pipeline, but every state moves through its own chain (prepare -> explore -> members -> screen -> early -> B -> guard -> A C D -> analysis) on ONE shared pool of pilot jobs, and is analysed (`run_analyses.py <exp> --tag --us never`, one at a time in the driver) as soon as its last run lands; the US sum and the overview come after every chain has ended. A later step outranks an earlier one on the pilots and, within a step, the bigger state goes first, so states finish one after another. Steps whose outputs exist are skipped (complete explore results file, members.csv, restart/screened.csv, restart/selection.csv; runs: only the missing ones), so finished states go straight to their analysis and resubmitting continues. Progress: `state_pipeline.py status --tag <tag>` or `analysis/<tag>/pipeline_status.csv`; with 13+ locations the overview `analysis/<tag>/index.html` is a status table (step, done time, B total/peak) linking each state's page, with only the US figures embedded. Analysis logs: `/scratch/$USER/flu-round1-2026-27/sw/drivers/<tag>-analysis/`. A pilot job that disappears is replaced (`top_up`); a task cut off mid-run stays Running on the server, so a state stuck with a few tasks left needs a resubmit.
* **Top-up to the guard** (it08, 2026-11-02): PPS keeps n proportionally only while the guard weights sum to >= n, so the pipeline (state_pipeline.py `--margin 1.25 --max-topups 2 --max-b 6`) checks `proj.guard_mass` after each B pass and, while the mass is under margin x keep-n, draws `proj.topup_size` more screened members with `proj.topup` -- from the screen selection's early-season weights, NOT from the guard's survivors (that would apply the guard twice and push out the high-burden parameter sets) -- runs their B, and guards once over every B run. New members continue the s#### ids with `topup_round` k (a MEMBER_META column); a guard selection already made is set aside as `restart/selection_round<k-1>.csv`. Each guard writes `analysis/<tag>/pass_rates.csv`; `--pass-rates` with an earlier tag's file sizes the first screen selection per state (margin x keep-n / pass rate, capped at max-b x keep-n). Preview-v3 pass rates (guard mass / B runs): MA 0.92 ... CA 0.68, LA 0.60, WA 0.35, FL 0.16.
* **Why FL and WA fail the guard** (2026-11-02): both rose early (FL flat ~110/wk = the 114 floor through August, then 146 -> 572 by Oct 3, x1.42/wk; WA 10 -> 137) and then SLOWED in October (hub data after the cutoff: FL 649, 875; WA 126, 156). The model's R(t) = tau x seasonal forcing can only grow in September with a high tau (FL screened tau_next_mult median 1.42 vs 0.98 in the candidates) and an early forcing peak, and that R keeps rising into winter: the B members pass near the Oct 3 level but grow x2.5-3/wk (FL B median 5,507 on Oct 17 vs 875 observed; 90% of members above it). None of FL's 128 candidates within 2x the Oct 3 level grows slower than x1.8/wk (CA: 23 near the level, growth 1.06-1.57: pass 0.68). In FL the faster-growing half of the screened members pass 0.00, the slower half 0.32. Not the guard, the floor or S0 (0.70-0.74 everywhere): the candidate pool cannot produce an early, slow rise. Options: a slope term in the early score (pre-cutoff data only), and a proposal that can (e.g. a next-season seasonal-amplitude draw, earlier/larger seeding for early states).
* **FL early-season test** (2026-11-02, tag `fl-early-test`, scenario B only): `slopex_FL` = prod2025o_FL's candidates and screen runs (copied) with the early score + a growth term (`proj._slope`: Gaussian on the last `early_slope_weeks` (4) weeks' log growth, sd `early_slope_sd` 0.1/week, weight `production.early_slope_weight`, default 0 = off); `seedx_FL` = model `flu-r1-2026-27-v3.esl` (a next-season A_H1N1 seeding pulse: `seed_boost_A_H1N1_next` x the fitted rate for `seed_days_A_H1N1_next` days from `seed_start_A_H1N1_next`; priors 1-30x log, 7-35 d, Aug 8 - Sep 12) + `escape_next_A_H1N1` 0-0.45, 8 fits x 32 draws. Both copy calib_prod2025o_FL (the explore pickle re-keyed to the new name). Run: `state_pipeline.py submit run slopex_FL seedx_FL --tag fl-early-test --per-fit 32 --no-acd --max-topups 0 --no-us --no-calib-report --jobs 24 --partition bii-largemem`; compare: `notebooks/fl_early_test.py` -> `analysis/fl-early-test/compare/` (also a seeding+growth estimate by re-weighting seedx's candidates). Re-weighting the preview's FL runs with the growth term alone moved the estimated pass rate only 0.14 -> 0.15 (weight 1) / 0.17 (weight 3): the candidate pool has no slow riser. The pipeline gained `--no-acd`, and chains of one state get their own label and analysis sub-tag (`<tag>/<experiment>`).
* **FL early-season results and the early-rise follow-up** (2026-11-02): pass rate baseline 0.16, growth term (sd 0.1) 0.20, seeding pulse 0.55; season total / record median 2.8 / 2.7 / 1.3x; Oct 24 model / NHSN (check only) 9.5 / 8.4 / 3.5x. The growth term at sd 0.1 was too weak (x1.6 vs x1.4 costs < 1 log-lik against NB level differences of tens); 0.03 is the noise of a 4-week log growth at 200-570/week. In seedx's B runs, those near the later data had tau_next_mult ~1.04, a Feb seasonal peak and x1.44/week to Oct 3; the overshooters tau ~1.26, a mid-Nov peak, x1.75 (pre-cutoff growth vs log Oct 24 overshoot r = 0.68). `seedx2_FL` = seedx_FL's candidates and screen runs with `early_slope_sd` 0.03. `notebooks/early_rise_scan.py` -> `analysis/early-rise-scan/`: the rise is national and H1N1 (80-97% of the last 4 weeks); the preview's pass rate falls with Oct 3 / record peak (FL 0.15, WA 0.18, LA 0.14, CA 0.10 vs < 0.05 elsewhere among the done states, passing 0.71-0.96); `fl_profile` = rising and >= 0.06 of the record (HI WA FL LA MT CA AL TN UT). `scripts/proj/make_seedx.py <ST>...` writes `seedx2_<ST>` (prod2025o_<ST> + the fix, round 1 copied; skips states whose preview explore is unfinished). Run: `state_pipeline.py submit run seedx2_FL seedx2_CA seedx2_WA seedx2_LA --tag early-rise --per-fit 32 --no-acd --max-topups 0 --no-us --no-calib-report --jobs 40 --partition bii-largemem`; the scan then adds `v_` columns and params.png / kept_params.csv (the kept members' seeding parameters, pooled and by state).
* **Preview** (`notebooks/production_preview.py prod2025n_VA ...` -> `calib_reports/<names>_preview.html`): the
  screened candidates, the selected members' season (total, peak week, H1N1 share) against last season and the
  early weeks, the members the guard dropped, by subtype. Run the pipeline small on the test states first
  (explore only, 8 fits x 16 draws, 60 screened, 30 kept by the guard, scenario B).
* **Run** the selected 300 in full for A-D (`flu_proj.py run --mode restart --selected`), then
  `smh_submission.py ... --selected`.
* **Cost** (all 50 locations, 40-core node-h): round 1 ~14 per point (2025-08 .. 2026-10), ~1.6k for 64 + 48
  points; screening ~3 per candidate, ~1.8k for 600; the full pass ~10 per member-scenario, ~12k for 300 x 4.
  ~15k node-h, ~7 days on 95 nodes.

```bash
python notebooks/informed_priors.py prod2025_* --suffix o --twostage --early-season --age-anchor --subtype-weekly 0.3 \
    --runs bt2023_VA bt2023_WA bt2023_MA                                    # -> prod2025o_<ST>
python scripts/calib/sw_calib.py submit explore  prod2025o_* --points 64 ...; ... optimize prod2025o_* ...
python scripts/proj/flu_proj.py members prod2025o_* --design sobol --top 20 --per-fit 30   # 600 candidates
python scripts/proj/flu_proj.py submit run prod2025o_* --mode screen --scenarios B --jobs 95
python scripts/proj/flu_proj.py select prod2025o_* --mode screen --weights early --n 900 --ess E
python scripts/proj/flu_proj.py submit run prod2025o_* --mode restart --selected --scenarios B --jobs 95
python scripts/proj/flu_proj.py select prod2025o_* --mode restart --weights guard --n 300
python scripts/proj/flu_proj.py submit run prod2025o_* --mode restart --selected --scenarios A C D --jobs 95
python notebooks/smh_submission.py prod2025o_* --selected
```

### Age re-anchoring of hospitalization (it08, `prod2025o_*`)

The contact network puts ~55% of the model's infections at 5-17 in every state and season, so the FluSurv-NET
IHR ladder (per infection) made 5-17 ~30% of admissions (NHSN 3-5%) and 65+ ~30% (NHSN 56-67%), in the
2024-25 and 2025-26 fits alike; the all-age likelihood never saw it. The early 2026-27 weeks showed it out of
sample (WA Sep 5 - Oct 3: 5-17 model 136 vs NHSN 6, 65+ 120 vs 291). The fix is Round 20's (outcome_probs.py):
`hosp_age_mult_<band>` (ESL configs, default 1) = NHSN admissions share / the fits' share per state (2024-25 +
2025-26, shrunk to the national mix with 200 pseudo-admissions), normalized to keep the all-age total
(`flu-round1-2026-27/notebooks/age_anchor.py` -> `data/processed/age_anchor/hosp_age_mult.csv`, `age_anchor.png`;
`informed_priors.py --age-anchor`). National: 0-4 0.58, 5-17 0.19, 18-49 0.91, 50-64 1.66, 65+ 1.68. Rescaling
the existing runs (admissions are a thinning of infections at I -> H, so the rescale is exact in expectation):
by-age NB of the top-20 fits -1700..-2090 -> -1010..-1580, all-age NB within +-40. Infections are untouched:
transmission is still child-driven (scenario C's indirect effect runs through it). Known limit: one multiplier
for all subtypes, from a pooled H1N1+H3N2 and an H3N2 season; H1N1 puts more admissions at 50-64.

### Several states at once on pilot jobs: `scripts/calib/sw_calib.py` (slurm-workflows)

[slurm-workflows](https://github.com/parantapa/slurm-workflows) (pilot jobs + a `ds-service` task server) and
[slurm-workflows-optimize](https://github.com/parantapa/slurm-workflows-optimize) (Sobol exploration, batch
Bayesian optimization) drive many experiments from one pool of 40-core bii nodes. Environment: conda env
`escape-sw` (a clone of `escape` + slurm-workflows 5.1, ds-service-client 7, slurm-workflows-optimize[botorch]);
server binary `~/bin/ds-service` 7.0.0 (2.2.0 kept in `~/bin/ds-service-versions/` for the old Round 20 scripts,
which use the slurm-workflows 1.x API in `escape`).

```bash
module load gcc/14.2.0 && conda activate escape-sw
python scripts/calib/sw_calib.py explore  bt2023_VA bt2023_WA bt2023_MA --points 64 --jobs 24   # Sobol, 64/state
python scripts/calib/sw_calib.py optimize bt2023_VA bt2023_WA bt2023_MA --parallelism 8 --jobs 24  # BO from there
python scripts/calib/sw_calib.py submit explore bt2023_VA bt2023_WA bt2023_MA --points 64 --jobs 24  # as a driver job
swtop <host:port printed by the driver>
```

* The objective is `flu_smh.sw.objective`: one run, scored with the same likelihood, returning
  `log(-fit_score)` (lower is better) plus every score field. Runs land in `calib_<name>/runs/sw<hash>.parquet`
  next to the particles.csv runs, so `flu_calib.py score`, `calib_plot` and the notebook see both.
* The run id hashes the parameters, so rerunning an interrupted `explore` (same seed) only simulates what is
  missing. `optimize` models every `calib_<name>/sw/*.pkl.gz` results file.
* **Report:** at the end, the driver runs `flu-round1-2026-27/notebooks/calib_report.py`, which writes
  `calib_<name>/report/index.html` per experiment (the `calibration_backtest.ipynb` figures and tables, plus
  search progress) and, for several experiments, a summary page. Rerun it any time with
  `python calib_report.py it04` (an iteration) or `python calib_report.py bt2023_VA bt2023_WA` (analysis venv or
  `escape-sw`), e.g. after a `flu_calib.py` array.
* **Iterations and report names.** `experiments/iterations.json` numbers the calibration rounds: it01-generic
  (bt2023), it02-informed (bt2023i), it03-lags-bg (bt2023j), it04-reduced (bt2023k + bt2024k), it05-production
  (prod2025k). Add a row (next id, short name, globs, one-line "about") whenever a new round of experiments is
  generated; never renumber. Summaries are named after what they cover: `calib_reports/it04-reduced.html`, part
  of one `it04-reduced_bt2024k.html`, across iterations `compare_it02_vs_it03.html` (figures in
  `calib_reports/<name>/`). `calib_reports/index.html` lists every iteration newest first with its pages and
  per-experiment reports; `latest.html` opens the newest one with runs. Pre-registry pages are in
  `calib_reports/old/`.
* The driver hosts the server on `ib0`, so it must run on a Rivanna node (login or compute), and stays up until
  every point is back. `submit` wraps it in `sw_calib.sbatch` (2 cores, 16 GB, 3 days), so each experiment's
  prepare (schedule, build, input.h5) runs as a task on a pilot node before any evaluation (`sw.prepare_task`;
  a state's input.h5 needs a whole node: AZ already exceeded 16 GB). Pilot logs:
  `/scratch/$USER/flu-round1-2026-27/sw/<stamp>-<cmd>/`.

### Production (`prod2025k_*`): all states, scenarios A-D, the hub file

The same procedure as the backtests, one year on, for every state with a synthetic population (49 + DC; AK has
none). Code: `flu-round1-2026-27/notebooks/prod_experiments.py` (base experiments), `informed_priors.py` (as for
the backtests), `py-modules/flu_smh/proj.py` + `scripts/proj/flu_proj.py` (members and paired runs on pilot jobs),
`flu-round1-2026-27/notebooks/smh_submission.py` (the hub parquet and its validation).

* **Person files.** `data/person/prod2025/` (tick 0 = 2025-08-03; 2022-23 to 2024-25 infections, 2023-24 and
  2024-25 campaigns): `run_person_parquets.py --anchor 2025-08-03 --vax-seasons 2023-24 2024-25 --inf-seasons
  2022-23 2023-24 2024-25 --tag prod2025`. Restart mode also needs the default production files
  `data/person/<ST>-person.parquet` (tick 0 = 2026-08-09, history through 2025-26): `run_person_parquets.py`.
* **Experiments.** `prod_experiments.py` writes `prod2025_<ST>` (anchor 2025-08-03, end 2027-06-05, fit
  2025-08-09..2026-06-27, projection window 2026-08-08..2027-06-05, season break 2026-08-02, campaigns 2025-26
  STAND-IN + 2026-27 = scenario B's Rd7 curve); then `informed_priors.py prod2025_* --suffix k --runs bt2023_VA
  bt2023_WA bt2023_MA` -> `prod2025k_<ST>`, the reduced design. `production` (experiment.py) holds the scenario
  map, the origin (2026-10-04), the restart anchor and person file, and `data_end` 2026-10-03: `calib.observed`
  drops NHSN after it, so the early-season weeks (Aug 8 - Oct 3) are scored and nothing later is seen.
* **Scenario campaigns.** `flu_data.COVERAGE["2026-27-A|B|C"]` (Rd7 curves on their own dates); a campaign
  `{"season": "2026-27", "coverage": key}` uses that curve, `"coverage": null` is none (D). The schedule is
  compiled in, so each scenario has its own build (`<exp>-<mode>-<letter>` on scratch) on one shared input.h5.
* **Calibrate** as for the backtests (`sw_calib.py submit explore|optimize prod2025k_* ...`); the calibration runs
  carry scenario B (the hub: calibrate to B).
* **Members.** `flu_proj.py members prod2025k_* --n 300 --top 20`: member k = the searched values of top run
  (k mod 20) + its own next-season draws (id m<k>); run_grouping = k + 1, shared by the four scenarios.
* **Modes.** Production uses **`restart`** (the default; decided 2026-10-27 on cost). `continuous`: the calibration's own two-season run with the scenario's 2026-27 campaign (what the
  backtests score; ~672 ticks; each scenario re-simulates 2025-26 with its own noise, since the simulator has no
  seed). `restart`: one season from 2026-08-09 on the production person file (NHSN-reconstructed history), the
  next-season block from tick 0 and absolute ticks shifted 371 d, `ltimm_frac_<sub>` re-solved per subtype so
  tick-0 S0 = the member's own calibration run's S0 by subtype on the origin, 2026-10-04 (`production.restart_s0`:
  "source" default, "calibration" = top-run median, or a number). Not the anchor: on Aug 9 the continuous runs still
  hold late-season infections in R, which drain by October. Validated on WY: restart vs continuous S0 0.681 vs
  0.680 (per subtype within 0.01), projected admissions +2%. ~300 ticks (~0.45x), and a
  pair shares its history exactly. Measured on WY: 135 s per restart run vs 385 s per two-season run (0.35x).
* **Run.** `flu_proj.py submit run prod2025k_* --mode restart --jobs 48 --qos bii-half` prepares every build on
  the workers (first scenario per state first: it builds the restart input.h5), then runs every missing
  (scenario, member), biggest states first; resumable. Outputs `proj_<exp>/<mode>/<scenario_id>/m####.parquet`
  (weekly infections / admissions / deaths by subtype x age) + `.json` (configs, S0 on 2026-10-04).
  `--extra-partition bii-largemem --extra-jobs N` adds a second job group on the same queue.
  Optional history weighting: run B for >= 2 x 300 members, `flu_proj.py select --n 300`, then A C D `--selected`.
* **Hub file.** `smh_submission.py prod2025k_* --mode restart [--age-groups] [--quantiles]` ->
  `submission/<team-model>/2026-10-04-<team-model>.gz.parquet`: inc hosp samples (horizon 1-35, weeks ending
  2026-10-10..2027-06-05), S0 + S0_A/B/AH1/AH3 tags (scenario B's value in all four, horizon NA), optional
  quantiles / peak size / peak time cdf, and US = per-member sum of the states (S0 population-weighted) when every
  jurisdiction ran. It validates against the `external/` hub clone's tasks.json and locations.csv
  (`--validate-only PATH` re-checks a file). Team-model is `UVA-escape_flu` (the default; the hub's
  `UVA-FluXSim` is a different model). The hub deadline (2026-10-20) has passed; the organizers know we are late.
* **Cost.** ~80% of a run's time is per tick, not per infection (bt2023j: run seconds = 1150-1480 fixed + ~0.02
  per admission). A two-season run costs ~84 s + ~221 s per million people on a 40-core node, so one member in one
  scenario in all 50 locations is ~22 node-h (continuous) or ~10 (restart); 300 members x 4 scenarios ~26k / ~12k
  node-h, against ~55 nodes under bii-half (+ ~43 idle 40-core bii-largemem nodes, whose QOS has no per-user cap).

## Still to do (snapshot from mid-October; superseded by AGENT_HANDOFF.md §7)

| area | status |
|---|---|
| person file | **done** for VA and WY: tick 0 = 2026-08-09, `last_infection_tick_A_H1N1/_A_H3N2/_B` (`run_person_parquets.py`, default `--infection-method subtype`). Other states: `run_initial_conditions.sh --slurm N` |
| prior vaccination history | the IC pipeline uses stand-in curves (Rd6 sc_A, Rd5 sc_C_D). B is now **2024-25** levels, not 2025-26, so it cannot stand in for the 2025-26 history. Real 2025-26 coverage still needs a source |
| scenario schedules | build A/B/C from the Rd7 curves × synthetic population (D empty), with `propensity_max` per band |
| seasonality | now a config cosine; fit `seas_peak_tick` / `seas_power` per state in calibration |
| subtypes | subtype-specific drift rates; fit the cross-protection strength (0.20 is a fixed assumption). Subtype-resolved history, subtype × age severity and A↔A cross-protection are done |
| S0 | the window-only S0 is ~0.80 in VA (~0.9+ with these kernel defaults) against the hub's 0.65–0.70. A background-immunity term is needed |
| severity | fit `hosp_prob_scale` (the ladder is observed rates that already include vaccination, so expect slightly above 1) and `hosp_prob_65p_scale`; a finer 65+ ladder (NHSN reports 65-74 and 75+); `death_prob_scale` against the CDC death target if deaths are submitted (it also counts out-of-hospital deaths) |
| drivers | the flu drivers are `py-modules/flu_smh` + `scripts/calib/`. The Round 20 `escape_covid19_smh` and `scripts/*.py` remain as reference only. Still needed: a production experiment (2025-26 fit → 2026-27 scenarios A–D) |
* **seedx2 results** (2026-11-04, tag `early-rise`, HI WA FL LA MT CA AL TN UT, B only): guard pass rate preview -> seedx2: FL 0.16 -> 0.60, WA 0.35 -> 0.50, UT 0.64 -> 0.83, AL 0.55 -> 0.78, MT 0.61 -> 0.73, CA 0.68 -> 0.68, LA 0.60 -> 0.58, TN 0.87 -> 0.84, HI 0.66 -> 0.48. The kept members fit Aug 8 - Oct 3 better (Oct 3 level 0.25-0.75x NHSN in the preview's kept sets of HI WA FL LA MT UT -> 0.8-1.5x; NB log-lik +4 to +23 in 6 states, -1 to -2 in CA AL TN; growth within ~0.1/week of observed except HI), but sit higher against the later weeks (check only; kept model / NHSN, Oct 10-24: FL 1.8 -> 2.6, LA 0.8 -> 1.9, MT 1.2 -> 2.7, HI 0.7 -> 1.7; WA 1.5 -> 0.85, TN 1.5 -> 1.1, UT 1.1 -> 0.6, CA/AL ~1): the preview's guard had kept members that started late, and the October slowdown (x1.1-1.3/week) is not in the fitting data. Pooled kept parameters: H1N1 pulse from ~Aug 31 (tick 387), ~6x (IQR 3-15) for ~25 days (19-31; WA, UT at the 35-day bound), escape_next_A_H1N1 ~0.24 (0.15-0.35), tau_next_mult ~0.87, seasonal peak ~Nov 23 (Nov 2 - Dec 28). `notebooks/early_rise_scan.py` writes kept.csv and kept_params.csv.
* **Validation truth from CDC** (2026-11-04): `flu_analysis.refresh_truth()` (and every `--refresh-truth`) now reads NHSN straight from data.cdc.gov (`ua7e-t2fy`, the weekly HRD by jurisdiction, plus weeks only the preliminary `mpgq-jmmr` has; `totalconfflunewadm` and the age columns, 65-130 = 65-74 + 75+) into `data/processed/validation/time-series-latest.csv`; `refresh_truth("hub")` keeps the old path. The hub's target data drops rows for up to 18 states a week, so the US sum (weeks where every state reported) was empty and fig11 had no dots; CDC has every state, KS and AR included. The fits still read the hub clone, cut at Oct 3.
* **preview-v4** (2026-11-04): preview-v3 with `seedx2_<ST>` for FL CA WA LA MT AL TN UT (HI stays prod2025o: seedx2 overshoots its Oct 3 level and its pass rate fell); the seedx2 states only need their A/C/D runs (30 kept x 3), the rest go straight to analysis. FL keeps 30 distinct members (15 in preview-v3, which capped the US at 15). `notebooks/early_rise_scan.py` adds kept.png (kept curves, preview vs seedx2) and the season columns of kept.csv.
* **FL alternative selection, preview-v4-alt** (2026-11-04): preview-v4's FL is 2.2-2.8x NHSN on Oct 10-24. NHSN slowed before the cutoff (x1.57, x1.37, x1.32/week over the last three weeks), and in the model the last week's growth predicts the next (r = 0.81 over seedx2_FL's screen runs; the 4-week growth -0.04). New early-score options: `production.early_recent_weeks` / `early_recent_weight` (`proj.recent_weights`: the last k weeks' NB terms, all-age and by subtype, weigh w x; default off) and the existing `early_slope_weeks`. Re-weighting seedx2_FL's 256 candidates moved the selection's next-week growth only x1.93 -> x1.76: about 6 of the 35 candidates near the Oct 3 level slow down, and they sit at seedx2's pulse bounds (boost ~23x, ~31 days ending ~Sep 20-25, tau_next_mult ~1, seasonal peak ~March). `seedx3_FL` = seedx2_FL + early_recent_weeks 3 x 3, the growth term over the last 2 weeks (sd 0.03), pulse priors 1-60x / 7-56 d, 8 fits x 64 draws (members.csv written). preview-v4-alt = preview-v4 with seedx3_FL for FL; `notebooks/fl_early_test.py` has it as the "recent-weighted" arm.
* **Deaths and severity replicates, preview-v4-up5** (2026-11-04): `flu_smh/upscale.py` turns each finished run into k trajectories by post-processing (no new simulation): replicate r of member m (`<m>r<r>`, r 0 = the run) keeps the run's infections and rescales its admissions by exp(national subtype + national age + state subtype log draws, sd 0.10 / 0.10 / 0.05), renormalized so scenario B's admissions to the fitting cutoff keep their total, times a weekly AR(1) noise (rho 0.8, sd 0.10, half national, half state) that starts at the cutoff. Admissions are scaled rather than recomputed from infections x IHR so that the model's protection against hospitalization (prior infection, vaccine) and the A-D contrasts are kept. Deaths of every trajectory = admissions x the model's own P(death | hospitalized) (`death_hosp_<sub>_<band>` x `death_prob_scale` parsed from the ESL/experiment), delayed by the ESL `hosp_to_death` kernel as week offsets (0.10, 0.70, 0.19), x a national log perturbation (sd from `death_given_hosp_by_subtype_age.csv` 95% CIs, floor 0.10); r 0's post-processed deaths match the simulated ones (CA 128.3k vs 129.0k summed over runs). Draws are keyed by member and replicate (never scenario), national ones by the member's sorted position, so US index pairing shares the national draw. `run_analyses.py --upscale K [--upscale-seed]` (writes `<tag>/upscale.json`); flu_analysis gained deaths in OUTCOMES (contrasts, fig06/07 third panel), `deaths_from_origin` in summary_all, fig12-deaths (vs past NHSN x the same ratios; US also the hub's CDC all-deaths estimate) and fig13-replicates (needs upscale). NOTE the model's P(death | hosp) is FluSurv-NET in-hospital (2.9% of admissions); the hub's target is CDC's all-deaths multiplier model, ~0.085 deaths per NHSN admission in 2024-25 (by age .018/.014/.033/.089/.111), i.e. ~2.7x the in-hospital count. smh_submission.py does not yet write deaths or replicates.
* **Replicate ids, observed deaths, growing to 60 runs** (2026-11-04): `upscale.Params.target` (N trajectories per state: ceil(N / runs) per run, dealt round by round, so 60 runs x 5 = 300 exactly and 55 runs -> 6 or 5 each); national draws now keyed by the trajectory's position in the sorted id list (the US sum's i-th sample shares them). `smh_submission.py --upscale K | --upscale-to N --deaths`: every row of a trajectory (inc hosp by age, inc death, S0) carries run_grouping = the run's run_grouping, stochastic_run = replicate + 1 (1 = as simulated); US sample i = the sum of the states' i-th trajectories, labelled run j / replicate r when every state has the same runs x replicates (else numbered); inc death (+ inc/cum death quantiles) US only; validate() also checks that every target and scenario of a location carries the same sample-id set. AK is not modelled, so the US needs `--national always` (flagged PARTIAL). Tested on preview-v4 (30 x 10): valid, 55 MB. fig12 adds NCHS provisional influenza deaths (death certificates, data.cdc.gov r8kw-7aab, `flu_analysis.refresh_nchs` on --refresh-truth -> data/processed/validation/nchs_flu_deaths.csv; NY = NY + NYC; weeks of 1-9 suppressed): US 2022-23..2025-26 9.4k / 10.0k / 18.2k / 10.6k, the in-hospital scale. Growing: `proj.grow_open` / `select(keep=)` / `state_pipeline.py --grow --keep-n 60 [--upscale-to 300]`: a state whose guard kept fewer reopens it (kept_grow.csv, selection_grow.csv), tops up B toward 1.25 x 60, keeps the old members for certain and PPS-draws the rest, so only new members run A C D (VA test: 30 kept + 30 drawn, 16 of them already had A C D from guard-v1). From preview-v4's B runs 13 states already have the mass, 37 need top-ups (~830 B runs), 5 have fewer than 60 passing so far. `--min-keep 45` (user: aim for 60, settle for 45): top-ups only to 1.25 x 45 guard mass, the guard keeps up to 60 of the passing runs. Run times from the runs' json (run_seconds): median 6 min, TX/CA 43, NY 33, PA/FL 22. Plan: ~100 top-up B + ~4,400 A/C/D runs, ~680 node-h, ~7 h on bii-largemem 44 + bii/bii-half 54 nodes.
* **Peak-timing guard, prod-60-v2** (2026-11-05): no state (50 + DC) peaked before November in NHSN 2022-26 (earliest SC, week ending 2022-11-12), but prod-60 kept 310 of 3,000 runs peaking (week ending) before Nov 1 (FL 35, LA 37, MT 31, HI 27, WA 17, TN 17). `proj.peak_ok` + `peak_after=` on guard_weights / guard_mass / topup_size / select (hard cut, weight 0); `member_totals` now has proj_model_peak_week; `grow_open(exp, n, peak_after)` also reopens a selection that keeps a failing run (survivors -> kept_grow.csv; old selection archived as selected_grow*.csv / selection_grow*.csv, never overwritten). `state_pipeline.py --peak-after 2026-11-01 --keep-n 60 --min-keep 45 --max-topups 2 --tag prod-60-v2`. Estimate from prod-60's B runs: ~610 top-up B (17 states; LA 172, FL 120, MT 72, HI 68, WA 46) + ~915 A/C/D = ~1,500 runs, ~260 node-h, ~3 h on 98 nodes; kept >= 57 everywhere. Cutoff Nov 8 would be ~2,700 runs / 470 node-h (min kept ~48); Nov 15 ~4,400 / 800 (min ~37). Passing runs pile up just after the cut (LA median peak week Nov 7).
* **All deaths, prod-60-v2 submission** (2026-11-05): the hub's `inc death` is CDC's in-season multiplier model (all flu deaths), ~4.9x the model's in-hospital deaths per admission. Post-processing only: `flu-round1-2026-27/notebooks/all_deaths_ratio.py` -> `data/processed/severity/all_deaths_per_admission.csv` (+ .json): level R = CDC in-season cumulative deaths (final week, geometric mean of the 95% bounds, Rd7_datasets/CumulativeDeaths_2022_2026.csv) / hub target-data NHSN US admissions over the same weeks, 2022-23..2025-26 = 0.154 / 0.177 / 0.110 / 0.136, centre 0.142, log-sd 0.203; age profile = CDC final burden deaths / hospitalizations (2022-25 geometric mean) normalized with NHSN 2024-26 age shares: q = .030 / .028 / .053 / .139 / .194 per admission (0-4 .. 65+), i.e. x5.4 / 5.2 / 3.4 / 4.8 / 5.0 the in-hospital ratio. `upscale.Params.all_deaths` + `DeathAnchor.from_model(..., all_table=)`: deaths x q[a] / p_pool[a] x exp(z_level + z_age[a]), national draws keyed by trajectory position on their own rng stream ("all_deaths"), drawn for r = 0 too; admissions and every other draw are bit-identical with and without it (VA check), A-D get the same factor. `run_analyses.py --all-deaths` (needs --upscale/--upscale-to; fig12 grey history lines then use q), `smh_submission.py --all-deaths`, abstract_results labels from upscale.json. CDC's final estimates have run below the in-season ones (0.083-0.116 per admission), so the centre follows what the hub scores against.
