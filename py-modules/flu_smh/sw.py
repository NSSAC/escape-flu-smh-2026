"""slurm-workflows integration: one experiment = one search space, one run = one objective evaluation.

The driver is scripts/calib/sw_calib.py. This module is what the pilot-job workers import (the driver puts
py-modules/ on their python path), so it holds only the objective and the space.

    space(exp)                     calib.space(exp) as slurm_workflows_optimize FloatRanges
    point_id(params)               deterministic run id for a parameter dict ("sw" + 10 hex)
    objective(experiment, **p)     run (or reuse) one parameter set, score it, return the result mapping

THE OBJECTIVE
  "objective"  log(-fit_score), lower is better. fit_score (calib.score_run) is a sum of log-probabilities,
               so -fit_score > 0. Its range spans ~400 (good) to 1e5+ (an explosive particle); the log keeps
               the Gaussian process of OptimizeSpaceBotorch from being dominated by the bad tail. The
               ranking is the same as fit_score's.
  everything else score_run reports (fit/projection NB log-likelihood, totals, peaks, subtype shares),
               plus run_id, host, run_seconds and reused.

RUN STORAGE AND RESUME
  A run writes <out>/runs/<run_id>.parquet (+ .stats.parquet, .json with its params) exactly as
  flu_calib.py does, so `flu_calib.py score` and calib_plot see slurm-workflows runs and particles.csv runs
  together. The run id is a hash of the parameters: re-running the same Sobol design (same seed) finds the
  finished runs and returns their scores without simulating, so an interrupted exploration resumes for free.

PER-WORKER STATE
  A worker evaluates many points of possibly several experiments. _context() caches each experiment's
  person-file age bands and NHSN observations for the life of the worker process.
"""
from __future__ import annotations

import hashlib
import json
import math
import socket
from functools import lru_cache

import pandas as pd

from . import calib, sim
from .experiment import Experiment, load


def space(exp: Experiment) -> dict:
    """FloatRanges; a mixture prior is searched as its unit coordinate (objective() maps it back)."""
    from slurm_workflows_optimize import FloatRange
    out = {}
    for k, spec in calib.space(exp).items():
        out[k] = FloatRange(0.0, 1.0) if calib.is_mix(spec) else FloatRange(spec[0], spec[1],
                                                                              log_range=(spec[2] == "log"))
    return out


def to_values(exp: Experiment, params: dict) -> dict:
    """Search coordinates -> parameter values (mixture priors arrive as unit coordinates)."""
    sp = calib.space(exp)
    return {k: float(calib.from_unit(sp[k], v)) if k in sp and calib.is_mix(sp[k]) else v
            for k, v in params.items()}


def point_id(params: dict) -> str:
    key = json.dumps({k: float(f"{float(v):.10g}") for k, v in sorted(params.items())}, sort_keys=True)
    return "sw" + hashlib.sha1(key.encode()).hexdigest()[:10]


@lru_cache(maxsize=None)
def _context(name: str):
    exp = load(name)
    band = sim.node_age_band(exp)
    obs = calib.observed(exp)
    mix_fit = calib.observed_subtype_mix(exp, exp.fit_start, exp.fit_end)
    mix_proj = calib.observed_subtype_mix(exp, exp.proj_start, exp.proj_end) if exp.proj_start else None
    return exp, band, obs, mix_fit, mix_proj


def objective(experiment: str, **params) -> dict:
    exp, band, obs, mix_fit, mix_proj = _context(experiment)
    params = to_values(exp, params)
    rid = point_id(params)
    out = calib.run_path(exp, rid)
    reused = out.exists() and out.with_suffix(".json").exists()
    if not reused:
        calib.run_params(exp, rid, params, band=band)
    meta = json.loads(out.with_suffix(".json").read_text())
    sc = calib.score_run(exp, pd.read_parquet(out), obs, mix_fit, mix_proj)
    return {"objective": math.log(max(-sc["fit_score"], 1e-9)), **sc, "run_id": rid,
            "host": socket.gethostname(), "run_seconds": meta["run_seconds"], "reused": reused}


def prepare_task(experiment: str) -> str:
    """sim.prepare on a pilot node (a whole node's memory: a big state's input.h5 needs far more than the
    driver has)."""
    exp = load(experiment)
    sim.prepare(exp)
    return str(sim.input_path(exp))
