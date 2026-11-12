# The vaccine in the model, and its realized effectiveness

How vaccination works in the flu model, why the vaccine was retuned in model v4 (Nov 2026), and how to measure
the effectiveness a parameter set produces. The tools are in the pipeline workspace:
`../flu-round1-2026-27/notebooks/ve_runs.py` (runs that keep per-person output) and `ve_analysis.py` (a matched
cohort, as in an observational VE study). Their docstrings have the full method.

## Mechanism (`models/flu-r1-2026-27-v4.esl`)

**One trivalent dose.** It protects against all three subtypes. Its strength is `vax_level(v)`:
```
vax_level = vax_waning_per_day ^ (t - last_vax_tick)                      # half-life 150 d
          × escape ^ (t - last_vax_formulation_tick)                      # antigenic drift since the formulation
escape    = vax_escape_per_day if >= 0, else immune_escape_per_day        # v4; 1 = no drift
```

**Three places the dose acts**, each scaled by `vax_level`:

| where | effect |
|---|---|
| susceptibility | × (1 − `vax_infection_ve` · L) |
| P(symptomatic \| infection) | × (1 − `vax_symp_ve` · L) (v4) |
| P(hospitalized \| infection) | × (1 − `vax_hosp_ve` · L) |

Hospitalization happens on the I -> H path, so the symptomatic term also protects against hospitalization.

**Who is vaccinated.** Doses come from a compiled-in schedule built from coverage curves by age band
(`flu_smh/schedule.py`). Within a band, the order follows the person file's `vax_propensity`, a Gaussian copula
correlated across seasons (about 77% repeat vaccination). `propensity_max` caps eligibility. Dose draws are random
in each run: an adult vaccinated in one run of a scenario is vaccinated in only about 81% of the other runs
(77% for 65+).

## Why v4 (Round 1 2026-27)

The hub specified 50% VE against medically attended illness and against hospitalization. That is an
*observed* VE, from test-negative designs. The model's inputs are per-agent efficacies, so the realized VE has to
be measured.
- **Measuring it.** Re-simulate members, keep their HDF5, and follow vaccinees against age-matched not-yet-vaccinated
  agents from 14 days after the dose. Estimate hazard VE per 30-day window and over the season.
- **v2 fell short.** v2's inputs (`vax_infection_ve` 0.25, `vax_hosp_ve` 0.3333, escape on) gave much less than the
  hub's 50%.
- **Not confounding or seeding.** The leaky effect under repeated exposure loses about 35-45% of the input on
  infection. It is not county or prior-vaccination confounding (`--strata county`, `county-prior`), and it is not
  seeding.

Measured on four VA scenario-B members (`analysis/ve/prod2025o_VA-B*`; median, with the range over members):

| | infection | symptomatic (medically attended proxy) | hospitalization |
|---|---|---|---|
| v2 inputs, realized | 0.10 (0.09-0.12) | 0.10 (0.10-0.12) | 0.33 (0.22-0.36) |
| v2 inputs, input-implied at event times | 0.18 | 0.18 | 0.37 |
| **v4** (`vax_escape_per_day` 1, `vax_infection_ve` 0.43, `vax_symp_ve` 0.46, `vax_hosp_ve` 0), realized | 0.23 (0.22-0.29) | **0.49** (0.45-0.53) | **0.52** (0.38-0.57) |

v4 meets the hub's 50% for both medically attended illness and hospitalization.
- **Why escape is off.** The hub's 50% is "directly applicable", including match, so drift on the vaccine is
  switched off.
- **The cost to the fit.** A stronger vaccine also removes transmission in the calibration season: 2025-26
  admissions fell to ×0.54 and peaks came 1-4 weeks later. Per infection, admissions fell only ×0.86.
- **The v3 shortcut** (prodv3, v3g and v5) rescaled rather than recalibrated: `tau_mult` 1.16 (baseline_tau and
  tau_seasonality_scale) and `hosp_prob_scale` ×1.08, from a VA sweep (`ve_runs.py plan-tau` / `sweep`).
- **What it leaves.** Uniform scaling restores the size of the season, not its shape. The proper fix, recalibrating
  under v4, is still open. A first try (prodv4) looked too damped.

## Measuring a new parameter set

```bash
cd ../flu-round1-2026-27/notebooks
python ve_runs.py prepare <exp> --scenario B --tag <tag> --model <model>.esl
python ve_runs.py plan <exp> --scenario B --tag <tag> --model <model>.esl --set name=value ...   # writes an sbatch
#   (the owner submits it)
python ve_analysis.py <exp> --scenario B --tag <tag>          # -> analysis/ve/<exp>-B-<tag>/
python ve_runs.py compare <exp> --scenario B --tag <tag>      # hospitalization level refit on the fit season
```

`plan` checks `--set` names against the ESL, because the simulator ignores unknown config names without warning.
Four members covering the spread of season sizes are enough to see the effect. Per-run 95% intervals come from
`--boot`.

## Related analyses

`detailed_analyses.py paired` measures the effect of a person's own dose on the outcomes they cause downstream.
Same people across scenarios, A/B/C vs D; VA, 3 members × 10 replicates:
- direct effect on their own hospitalization: about −37%;
- on causing at least one hospitalization at generations 1, 2, 3 and 4-10: about −24% at each.
