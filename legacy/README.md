# legacy/

Inherited from [NSSAC/escape-covid19-smh](https://github.com/NSSAC/escape-covid19-smh) (COVID-19 Round 20) and
[NSSAC/escape-cryptic-smh](https://github.com/NSSAC/escape-cryptic-smh) (cryptic round, SPOK-24). Nothing in the
flu model or drivers uses these files. They are kept as reference for where the flu model's design came from.
They are not maintained, and their paths point at covid19-round20, cryptic-round and other users' home
directories.

| path | what |
|---|---|
| `models/covid19-r20.esl` | the Round 20 COVID model (per-agent immune clocks, waning, escape, vaccination): the starting point of `models/flu-r1-2026-27.esl` |
| `models/model.esl` | an earlier Round 20 version |
| `models/spok24.esl` | the SPOK-24 cryptic-round model (naive population, compiled-in importation, admission counting) |
| `py-modules/escape_covid19_smh/` | the Round 20 wrapper package (`escape_wrapper`, `calibrate`, `compute_seasonality`, `read_data`, `spok24`) |
| `scripts/` | Round 20 and SPOK-24 drivers (`run_projection.py`, `explore_params.py`, `opt_va.py`, `run_all_states_scnB.py`, `test_*.py`) |
| `docs/` | the Round 20 calibration-and-projection process and round notes |

Moved here on 2026-11-12 (from `models/`, `py-modules/`, `scripts/`, `docs/`).
