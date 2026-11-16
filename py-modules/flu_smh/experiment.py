"""An experiment: everything that defines a family of runs, in one JSON file (experiments/<name>.json).

    exp = load("bt2023_VA")
    exp.tick("2024-01-13")        # date -> tick
    exp.num_ticks                 # anchor .. end inclusive

Fields
  name, state            experiment id; USPS code
  anchor                 tick 0 (a Sunday). The person file must be built with this anchor.
  end                    last simulated day
  fit_start, fit_end     week-ending dates scored against NHSN (calibration window)
  proj_start, proj_end   week-ending dates of the out-of-sample window (backtest skill)
  season_break           date from which the model's *_next season block applies (fit season ->
                         projected season); null = never
  person_file            path to <ST>-person.parquet (relative paths resolve against FLU_ROUND)
  network_file           EpiHiper contact network (default: the state's v2.4.0 network)
  campaigns              in-sim vaccination: [{"season": "2023-24", "campaign": 1}, ...]; seasons are
                         keys of flu_data.COVERAGE
  propensity_slack       eligible pool per band = the lowest-propensity agents covering
                         min(slack x final coverage, 0.999) of the band (1 = exactly the habitual
                         vaccinators; large = random)
  fixed                  config values held fixed for every particle
  priors                 {config: [lo, hi, "lin"|"log"]} or a mixture {"mix": [[lo, hi, weight], ...],
                         "scale": "lin"|"log"} (segments in increasing order; sampled by inverse CDF from
                         a unit coordinate, so a segment's weight is its prior probability). Also
                         "seed_w_<sub>" weights normalized into seed_share_<sub>, and "share_w_<sub>" target
                         subtype weights mapped to tau_scale_<sub> through share_map (calib.configs_for)
  draws                  {config: prior} like priors, but drawn per run instead of searched (calib DRAWS):
                         the next-season parameters the fit cannot see. "tau_next_mult" scales the three
                         next-season tau_scale_* (the projected season's intensity)
  share_map              {"fit"|"next": {"A_H3N2"|"B": {"c", "A_H3N2", "B", "seed"}}}: the fitted
                         log(share_k / share_H1) = c + b . log(tau_scale / tau_scale_H1) + seed . log(seed_w_k /
                         seed_w_H1) that turns share_w_* into tau_scale_* (informed_priors.py)
  model                  ESL file under models/ (default flu-r1-2026-27.esl; v2 adds per-subtype seasonal
                         lags and a next-season peak)
  next_season            "sample" (independent draws of the *_next subtype parameters: the next
                         season's mix is unknown) or "same" (carry the fit season's values over)
  n_particles, seed      Sobol sample size and scramble seed
  likelihood             {"nb_size": k, "subtype_weight": w, "subtype_neff": n} (+ background, background_mode,
                         optional week_weight; see calib.py LIKELIHOOD)
  keep_h5                keep each run's raw HDF5 (large) or delete it after extraction
  root, scratch          optional: the study directory this experiment's outputs (calib_<name>, proj_<name>) and
                         relative person files live under, and the scratch directory of its builds. Default:
                         FLU_ROUND / FLU_SCRATCH from the environment. Set by studies other than Round 1, so a
                         worker started without the study's environment still writes into the right tree
                         (${USER} etc. are expanded)
  production             hub projections from this calibration (flu_smh/proj.py), or {} for a backtest:
                           season          the campaign the scenarios replace ("2026-27")
                           origin          hub origin_date, the first projected day; S0 is read here
                           scenarios       {scenario_id: flu_data.COVERAGE key, or null for no campaign}
                           restart_anchor, restart_person_file
                           screen_end      optional: the last day of a mode-"screen" run (default origin + 7 d)
                                           mode "restart": a single-season run from this tick 0 and person
                                           file (history to the anchor) instead of the continuous run
"""
from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FLU_ROUND = Path(os.environ.get("FLU_ROUND", "/project/bii_nssac/biocomplexity/SMH/flu-round1-2026-27"))
EXPERIMENTS = REPO / "experiments"
MODEL = REPO / "models" / "flu-r1-2026-27.esl"
SCRATCH = Path(os.environ.get("FLU_SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}/flu-round1-2026-27/calib"))
NETWORK_FMT = "/project/bii_nssac/EpiHiperSynPop/v2.4.0/from-alt-project/usa_{st}/{st}_contact_network.csv"

SUBTYPES = ["A_H1N1", "A_H3N2", "B"]


@dataclass
class Experiment:
    name: str
    state: str
    anchor: str
    end: str
    fit_start: str
    fit_end: str
    proj_start: str | None
    proj_end: str | None
    season_break: str | None
    person_file: str
    campaigns: list
    priors: dict
    fixed: dict = field(default_factory=dict)
    network_file: str | None = None
    propensity_slack: float = 1.25
    next_season: str = "sample"
    n_particles: int = 64
    seed: int = 2023
    likelihood: dict = field(default_factory=lambda: {"nb_size": 20.0, "subtype_weight": 1.0,
                                                      "subtype_neff": 200.0})
    keep_h5: bool = False
    notes: str = ""
    model: str = "flu-r1-2026-27.esl"
    share_map: dict = field(default_factory=dict)
    draws: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    production: dict = field(default_factory=dict)
    root: str | None = None
    scratch: str | None = None

    # --- derived ---------------------------------------------------------------------------
    @property
    def t0(self) -> pd.Timestamp:
        return pd.Timestamp(self.anchor)

    def tick(self, date) -> int:
        return int((pd.Timestamp(date) - self.t0).days)

    def date(self, tick: int) -> pd.Timestamp:
        return self.t0 + pd.Timedelta(days=int(tick))

    @property
    def num_ticks(self) -> int:
        return self.tick(self.end) + 1

    @property
    def season_break_tick(self) -> float:
        return float(self.tick(self.season_break)) if self.season_break else 1e9

    @property
    def home(self) -> Path:
        """The study directory: `root` if the experiment names one, else FLU_ROUND."""
        return Path(os.path.expandvars(self.root)) if self.root else FLU_ROUND

    @property
    def person_path(self) -> Path:
        p = Path(self.person_file)
        return p if p.is_absolute() else self.home / p

    @property
    def network_path(self) -> Path:
        return Path(self.network_file or NETWORK_FMT.format(st=self.state.lower()))

    @property
    def model_path(self) -> Path:
        p = Path(self.model)
        return p if p.is_absolute() else REPO / "models" / p

    @property
    def out_dir(self) -> Path:
        """Small results (particles, weekly tables, scores) -- under flu-round1-2026-27/calib_<name>."""
        return self.home / f"calib_{self.name}"

    @property
    def work_dir(self) -> Path:
        """Build and input.h5 (large) -- on scratch."""
        return (Path(os.path.expandvars(self.scratch)) if self.scratch else SCRATCH) / self.name

    def base_configs(self) -> dict:
        """Configs every particle shares: the fixed values plus the experiment's calendar."""
        return {**self.fixed, "season_break_tick": self.season_break_tick}


def load(name_or_path: str) -> Experiment:
    p = Path(name_or_path)
    if not p.suffix:
        p = EXPERIMENTS / f"{name_or_path}.json"
    d = json.loads(p.read_text())
    d.pop("_comment", None)
    return Experiment(**d)


# --------------------------------------------------------------------------------------
# calibration iterations (experiments/iterations.json)
# --------------------------------------------------------------------------------------

ITERATIONS = EXPERIMENTS / "iterations.json"


def iterations() -> list[dict]:
    """The registry, oldest first. Each row adds "tag" (<id>-<name>, e.g. it04-reduced) and "members" (the
    experiment names its globs match; experiment files only, so a typo shows up as an empty iteration)."""
    if not ITERATIONS.exists():
        return []
    rows = json.loads(ITERATIONS.read_text())["iterations"]
    names = sorted(p.stem for p in EXPERIMENTS.glob("*.json") if p != ITERATIONS)
    return [{**it, "tag": f"{it['id']}-{it['name']}",
             "members": [n for n in names if any(fnmatch(n, g) for g in it["experiments"])]} for it in rows]


def iteration_of(name: str) -> dict | None:
    """The iteration an experiment belongs to (None if unregistered)."""
    return next((it for it in iterations() if name in it["members"]), None)
