"""Particles, runs and scores for an Experiment.

    space(exp)             {parameter: prior spec}: exp.priors plus the automatic *_next dimensions (searched)
    draw_space(exp)        {parameter: prior spec} of exp.draws: drawn per run, NOT searched (see DRAWS)
    draws_for(exp, id)     one run's draws, reproducible from the run id
    from_unit(spec, u)     a unit coordinate -> the parameter value (inverse CDF; mixtures included)
    sample(exp)            Sobol particles over space(exp) -> <out>/particles.csv
    configs_for(exp, row)  one particle -> the ESL configs (fixed values, calendar, derived seed shares,
                           the next-season block)
    run_particle(exp, i)   simulate particle i of particles.csv -> <out>/runs/p<i>.parquet (+ stats, json)
    run_params(exp, id, p) the same for any parameter dict under a string id (slurm-workflows runs,
                           flu_smh/sw.py) -> <out>/runs/<id>.parquet; the json records the params
    observed(exp)          NHSN weekly admissions for the state (all ages; NHSN bands where reported)
    score(exp)             negative-binomial log-likelihood over the fit window (+ subtype-mix term),
                           projection-window skill -> <out>/scores.csv
    ensemble_skill(...)    the projection scored as an ensemble of the best-fitting runs (WIS, interval
                           coverage, season total and peak against their spread)

DRAWS
  Next-season parameters (the projected season's timing, mix, seeding and intensity) cannot be seen by the
  fit-season likelihood, so searching them only adds dimensions the optimizer cannot resolve. exp.draws holds
  their priors; each run draws its own values (seeded by the experiment seed and the run id, so a re-run
  draws the same), and the run JSON records them with the searched parameters. The top runs' projections
  then form a prior-predictive ensemble over the next season.

LIKELIHOOD
  admissions   sum over fit weeks of log NB(obs | mean = model all-age admissions with background, size =
               nb_size). NHSN 2023-24 has no age bands (they start 2024-11), so the fit is all-age; ages
               appear in the projection window only.
  background   likelihood["background"] = {"fit": b, "proj": b} admissions per week, and
               likelihood["background_mode"]: "floor" (mean = max(model, b), the default for new
               experiments) or "add" (model + b; bt2023j, where a constant b also counted in-season, ~30% of
               WA's season, letting the model fit with a 6% attack rate). It stands for the sporadic off-season
               admissions (NHSN summer floor 9-30 a week) that the model,
               whose transmission dies out between seasons, does not simulate. Without it a single week with
               model ~0 and 10 observed costs ~150, more than most misfits at the peak.
  week weight  optional likelihood["week_weight"] = {"threshold": 0.1, "low": 0.25}: weeks observed below
               10% of the window's peak count 0.25 in the NB sum (fit and projection windows alike). With a
               fixed nb_size the NB is already a relative error in season (CV floor 1/sqrt(nb_size)), so this
               only shifts weight off the tails and off-season; on the it05 explore it changed 0-2 of the top 10.
  subtype mix  subtype_weight x subtype_neff x sum_k p_obs,k log p_model,k (a multinomial with an
               effective sample size). p_obs = the season's infection mix in the state from the
               NHSN/IHR reconstruction (flu-round1-2026-27 infections_weekly.parquet), p_model = the
               run's. This identifies tau_scale and seed_share by subtype, which admissions alone do not.
  by subtype   optional likelihood["subtype_weekly"] = w: adds w x sum_k NB(NHSN x lab share_k | model
               admissions_k) over the weeks (fit_subtype_nb_ll; the background floor split by the week's
               share). The season-total mix term cannot see WHEN each subtype peaks; this can, which is
               what identifies the seasonal lags. calib.score always reports it (diagnostics).
  projection   reported, not used for ranking: the same NB log-likelihood and the ratio of season totals
               and peaks, over proj_start .. proj_end.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import qmc

from . import sim
from .experiment import FLU_ROUND, SUBTYPES, Experiment

sys.path.insert(0, str(FLU_ROUND / "notebooks"))


# --------------------------------------------------------------------------------------
# particles
# --------------------------------------------------------------------------------------

def _scale(u: np.ndarray, lo: float, hi: float, kind: str) -> np.ndarray:
    if kind == "log":
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


def _spec(v) -> tuple:
    """A prior as written in the experiment -> (lo, hi, "lin"|"log") or ("mix", segments, scale)."""
    if isinstance(v, dict):
        seg = [(float(a), float(b), float(w)) for a, b, w in v["mix"]]
        return ("mix", seg, v.get("scale", "lin"))
    return (float(v[0]), float(v[1]), v[2])


def is_mix(spec) -> bool:
    return spec[0] == "mix"


def from_unit(spec, u):
    """Unit coordinate(s) -> value(s). A mixture is sampled by inverse CDF: segment i owns a slice of
    [0, 1] proportional to its weight, and maps that slice onto [lo_i, hi_i] (on the prior's scale)."""
    u = np.asarray(u, dtype=float)
    if not is_mix(spec):
        return _scale(u, *spec)
    _, seg, kind = spec
    w = np.array([s[2] for s in seg], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(w / w.sum())])
    i = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(seg) - 1)
    lo = np.array([s[0] for s in seg])[i]
    hi = np.array([s[1] for s in seg])[i]
    frac = np.where(edges[i + 1] > edges[i], (u - edges[i]) / np.maximum(edges[i + 1] - edges[i], 1e-12), 0.5)
    return _scale(np.clip(frac, 0, 1), lo, hi, kind)


def space(exp: Experiment) -> dict:
    """The sampled parameters and their prior specs. With next_season="sample", each fit-season subtype
    parameter (tau_scale_*, seed_w_*) without its own *_next prior or fixed value gets an independent *_next
    dimension with the same range."""
    sp = {k: _spec(v) for k, v in exp.priors.items()}
    if exp.next_season == "sample":
        for k in list(exp.priors):
            nk = f"{k}_next"
            if k.startswith(("tau_scale_", "seed_w_")) and not k.endswith("_next") \
                    and nk not in sp and nk not in exp.fixed:
                sp[nk] = sp[k]
    return sp


def draw_space(exp: Experiment) -> dict:
    return {k: _spec(v) for k, v in exp.draws.items()}


def draws_for(exp: Experiment, rid: str) -> dict:
    """The run's next-season draws: independent uniforms through each prior's inverse CDF."""
    sp = draw_space(exp)
    if not sp:
        return {}
    rng = np.random.default_rng([int(exp.seed), zlib.crc32(str(rid).encode())])
    return {k: float(from_unit(spec, rng.random())) for k, spec in sp.items()}


def sample(exp: Experiment, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Scrambled Sobol over space(exp)."""
    n = n or exp.n_particles
    sp = space(exp)
    u = qmc.Sobol(d=len(sp), scramble=True, seed=seed if seed is not None else exp.seed).random(n)
    df = pd.DataFrame({"particle": np.arange(n)})
    for j, (name, spec) in enumerate(sp.items()):
        df[name] = from_unit(spec, u[:, j])
    return df


def configs_for(exp: Experiment, row: pd.Series | dict) -> dict:
    row = dict(row)
    cfg = exp.base_configs()
    for k in [k for k in cfg if k.startswith(("seed_w_", "share_w_"))]:     # fixed weights: inputs, not configs
        row.setdefault(k, cfg.pop(k))
    for k, v in row.items():
        if k == "particle" or k.startswith(("seed_w_", "share_w_")):
            continue
        cfg[k] = float(v)
    for suffix in ("", "_next"):
        w = {s: row.get(f"seed_w_{s}{suffix}") for s in SUBTYPES}
        if suffix == "_next" and exp.next_season == "same":
            w = {s: row.get(f"seed_w_{s}") for s in SUBTYPES}
        if all(v is not None for v in w.values()):
            tot = sum(w.values())
            for s in SUBTYPES:
                cfg[f"seed_share_{s}{suffix}"] = w[s] / tot
    if exp.next_season == "same":
        for s in SUBTYPES:
            if f"tau_scale_{s}" in cfg:
                cfg[f"tau_scale_{s}_next"] = cfg[f"tau_scale_{s}"]
    for suffix, block in (("", "fit"), ("_next", "next")):
        if all(f"share_w_{s}{suffix}" in row for s in SUBTYPES):
            cfg |= tau_from_shares(exp, row, suffix, block, cfg)
    if "tau_next_mult" in cfg:                    # next-season intensity: scales all three next-season taus
        m = cfg.pop("tau_next_mult")
        for s in SUBTYPES:
            cfg[f"tau_scale_{s}_next"] = cfg.get(f"tau_scale_{s}_next", 1.0) * m
    if "tau_mult" in cfg:                         # transmission x m in both seasons, every subtype (an experiment's
        m = cfg.pop("tau_mult")                   # fixed value scales its members' searched taus; prodv3_*, 2026-11)
        for k in ("baseline_tau", "tau_seasonality_scale"):
            if k in cfg:
                cfg[k] = cfg[k] * m
    if "ltimm_frac" in cfg:                       # one long-term-immunity fraction for all subtypes
        f = cfg.pop("ltimm_frac")
        for s in SUBTYPES:
            cfg.setdefault(f"ltimm_frac_{s}", f)
    if "infection_half_life_days" in cfg:         # sampled as a half-life, passed as a per-day factor
        cfg["infection_waning_per_day"] = 0.5 ** (1.0 / cfg.pop("infection_half_life_days"))
    return cfg


def tau_from_shares(exp: Experiment, row: dict, suffix: str, block: str, cfg: dict) -> dict:
    """Target subtype weights share_w_<sub><suffix> -> tau_scale_<sub><suffix> for all three subtypes, by
    inverting the fitted share map (which gives log tau ratios to H1) and normalizing the three to a
    geometric mean of 1: the mix moves, the overall level stays with baseline_tau and the seasonal forcing
    (otherwise an "absent H1" draw would inflate the whole season). A reparameterization for the prior:
    the run's realized mix still comes from the model, and the likelihood scores it."""
    m = exp.share_map[block]
    w = {s: max(float(row[f"share_w_{s}{suffix}"]), 1e-6) for s in SUBTYPES}
    seed = {s: row.get(f"seed_w_{s}{suffix}", row.get(f"seed_w_{s}")) for s in SUBTYPES}
    others = ["A_H3N2", "B"]
    y = []
    for k in others:
        ls = np.log(seed[k] / seed["A_H1N1"]) if seed[k] is not None and seed["A_H1N1"] is not None else 0.0
        y.append(np.log(w[k] / w["A_H1N1"]) - m[k]["c"] - m[k]["seed"] * ls)
    A = np.array([[m[k][j] for j in others] for k in others], dtype=float)
    lt = dict(zip(["A_H1N1", *others], [0.0, *np.linalg.solve(A, np.array(y))]))
    mean = sum(lt.values()) / len(lt)
    return {f"tau_scale_{k}{suffix}": float(np.clip(np.exp(v - mean), 0.3, 3.0)) for k, v in lt.items()}


# --------------------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------------------

def run_id(i: int | str) -> str:
    """particles.csv rows are p0000, p0001, ...; slurm-workflows runs carry their own string id."""
    return f"p{i:04d}" if isinstance(i, (int, np.integer)) else str(i)


def run_path(exp: Experiment, i: int | str) -> Path:
    return exp.out_dir / "runs" / f"{run_id(i)}.parquet"


def run_particle(exp: Experiment, i: int, particles: pd.DataFrame | None = None,
                 band: np.ndarray | None = None, verbose: bool = False) -> Path:
    particles = particles if particles is not None else pd.read_csv(exp.out_dir / "particles.csv")
    row = particles.set_index("particle").loc[i]
    return run_params(exp, i, row.to_dict(), band=band, verbose=verbose)


def run_params(exp: Experiment, i: int | str, params: dict, band: np.ndarray | None = None,
               verbose: bool = False) -> Path:
    """Simulate one parameter set (keys of space(exp)) plus the run's draws and write its weekly table, stats
    and json (whose params hold both)."""
    rid = run_id(i)
    params = {**draws_for(exp, rid), **{k: v for k, v in params.items() if k != "particle"}}
    cfg = configs_for(exp, params)
    h5 = exp.work_dir / "runs" / f"{rid}.h5"
    t0 = time.time()
    sim.run(exp, cfg, h5, verbose=verbose)
    t_run = time.time() - t0
    w = sim.extract(exp, h5, band)
    st = sim.stats(exp, h5)
    out = run_path(exp, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.assign(particle=rid).to_parquet(out, index=False)
    st.assign(particle=rid).to_parquet(out.with_suffix(".stats.parquet"), index=False)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"particle": rid, "params": {k: float(v) for k, v in params.items() if k != "particle"},
         "configs": cfg, "run_seconds": round(t_run, 1),
         "extract_seconds": round(time.time() - t0 - t_run, 1)}, indent=1))
    if not exp.keep_h5:
        h5.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------------------
# data and scores
# --------------------------------------------------------------------------------------

def observed(exp: Experiment) -> pd.DataFrame:
    """NHSN weekly admissions for exp.state: date, age_group ('0-130' and bands where reported), value. A
    production experiment sees nothing after production["data_end"] (the hub's fitting-data cutoff)."""
    import surveillance_data as sd
    h = sd.load_nhsn(start=str(exp.t0.date()))
    h = h[h.location == exp.state]
    if exp.production.get("data_end"):
        h = h[h.date <= pd.Timestamp(exp.production["data_end"])]
    return h[["date", "age_group", "value"]].reset_index(drop=True)


def observed_subtype_mix(exp: Experiment, start, end) -> pd.Series:
    """Infection share by subtype in [start, end] for the state, from the NHSN/IHR reconstruction."""
    import flu_data as fd
    f = fd.PROCESSED / "infections" / "infections_weekly.parquet"
    inf = pd.read_parquet(f)
    x = inf[(inf.location == exp.state) & (inf.date >= pd.Timestamp(start)) & (inf.date <= pd.Timestamp(end))]
    s = x.groupby("subtype").infections.sum().reindex(SUBTYPES)
    return s / s.sum()


@lru_cache(maxsize=None)
def _lab_shares(state: str) -> pd.DataFrame:
    import flu_data as fd
    sh = pd.read_csv(fd.PROCESSED / "infections" / "subtype_shares_weekly.csv", parse_dates=["date"])
    return sh[sh.location == state].set_index("date")[list(SUBTYPES)]


def observed_by_subtype(exp: Experiment, o: pd.Series) -> pd.DataFrame:
    """NHSN all-age admissions o (date-indexed) split by the week's lab subtype share (the shares the
    infection reconstruction uses; NHSN itself reports no subtype): date x subtype."""
    return _lab_shares(exp.state).reindex(o.index).mul(o, axis=0)


def nb_loglik(obs: np.ndarray, mu: np.ndarray, k: float, w: np.ndarray | float = 1.0) -> float:
    mu = np.maximum(mu, 1e-6)
    return float(np.sum(w * (gammaln(obs + k) - gammaln(k) - gammaln(obs + 1)
                             + k * np.log(k / (k + mu)) + obs * np.log(mu / (k + mu)))))


def week_weights(exp: Experiment, obs: np.ndarray) -> np.ndarray | float:
    """likelihood["week_weight"] = {"threshold": f, "low": w}: weeks whose observed admissions are below
    f x the window's observed peak count w (< 1) in the NB sum; in-season weeks count 1. Absent -> 1."""
    ww = exp.likelihood.get("week_weight")
    if not ww or not len(obs):
        return 1.0
    return np.where(obs >= ww.get("threshold", 0.1) * obs.max(), 1.0, ww.get("low", 0.25))


def background(exp: Experiment, win: str) -> float:
    """Background admissions per week in window `win` ("fit" | "proj")."""
    return float(exp.likelihood.get("background", {}).get(win, 0.0))


def with_background(exp: Experiment, win: str, x):
    """Model admissions -> the likelihood's mean: max(x, b) ("floor") or x + b ("add", the bt2023j form)."""
    b = background(exp, win)
    return np.maximum(x, b) if exp.likelihood.get("background_mode", "add") == "floor" else x + b


def subtype_nb_ll(exp: Experiment, win: str, oo: pd.Series, ms: pd.DataFrame,
                  extra: np.ndarray | float = 1.0) -> float:
    """sum_k NB(NHSN x lab share_k | model admissions_k) over the weeks of oo (observed all-age admissions,
    date-indexed); ms = the run's admissions, date x subtype. The background floor is split by the week's share.
    extra: further per-week weights over oo's weeks (proj.recent_weights)."""
    lk = exp.likelihood
    os_ = observed_by_subtype(exp, oo).dropna()
    bg = background(exp, win)
    ll = 0.0
    for s in SUBTYPES:
        ms_s = ms.get(s, pd.Series(0.0, index=ms.index)).reindex(os_.index).fillna(0.0).to_numpy()
        share = (os_[s] / oo.reindex(os_.index).replace(0, np.nan)).fillna(1 / 3).to_numpy()
        mu = np.maximum(ms_s, bg * share) if lk.get("background_mode", "add") == "floor" else ms_s + bg * share
        xw = pd.Series(extra, index=oo.index).reindex(os_.index).to_numpy() if np.ndim(extra) else extra
        ll += nb_loglik(os_[s].to_numpy(), mu, lk["nb_size"],
                        week_weights(exp, oo.reindex(os_.index).to_numpy()) * xw)
    return ll


def score_run(exp: Experiment, w: pd.DataFrame, obs: pd.DataFrame, mix_fit: pd.Series,
              mix_proj: pd.Series | None, diagnostics: bool = False) -> dict:
    """One run's scores. The weekly-by-subtype NB term ({win}_subtype_nb_ll) is computed when the
    experiment weights it (likelihood["subtype_weekly"]) or for diagnostics (calib.score, the reports);
    otherwise the keys match older runs (slurm-workflows records)."""
    lk = exp.likelihood
    m = w.groupby("date").admissions.sum()
    by_sub = diagnostics or "subtype_weekly" in lk
    ms = w.pivot_table(index="date", columns="subtype", values="admissions", aggfunc="sum") if by_sub else None
    o = obs[obs.age_group == "0-130"].set_index("date").value
    out = {}
    for win, (a, b) in {"fit": (exp.fit_start, exp.fit_end), "proj": (exp.proj_start, exp.proj_end)}.items():
        if a is None:
            continue
        oo = o[(o.index >= pd.Timestamp(a)) & (o.index <= pd.Timestamp(b))].dropna()
        mm = m.reindex(oo.index).fillna(0.0)
        out[f"{win}_nb_ll"] = nb_loglik(oo.to_numpy(), with_background(exp, win, mm.to_numpy()), lk["nb_size"],
                                        week_weights(exp, oo.to_numpy()))
        if by_sub:
            out[f"{win}_subtype_nb_ll"] = subtype_nb_ll(exp, win, oo, ms)
        out[f"{win}_weeks"] = int(len(oo))
        out[f"{win}_obs_total"] = float(oo.sum())
        out[f"{win}_model_total"] = float(mm.sum())
        out[f"{win}_obs_peak_week"] = oo.idxmax().date().isoformat() if len(oo) else None
        out[f"{win}_model_peak_week"] = mm.idxmax().date().isoformat() if len(mm) else None
        out[f"{win}_obs_peak"] = float(oo.max()) if len(oo) else np.nan
        out[f"{win}_model_peak"] = float(mm.max()) if len(mm) else np.nan
        mix = mix_fit if win == "fit" else mix_proj
        if mix is not None:
            ww = w[(w.date >= pd.Timestamp(a)) & (w.date <= pd.Timestamp(b))]
            pm = ww.groupby("subtype").infections.sum().reindex(SUBTYPES)
            pm = (pm / pm.sum()).clip(lower=1e-6)
            out[f"{win}_subtype_ll"] = float(lk["subtype_neff"] * np.sum(mix * np.log(pm)))
            for s in SUBTYPES:
                out[f"{win}_model_share_{s}"] = float(pm[s])
                out[f"{win}_obs_share_{s}"] = float(mix[s])
    out["fit_score"] = out["fit_nb_ll"] + lk["subtype_weight"] * out.get("fit_subtype_ll", 0.0) \
        + lk.get("subtype_weekly", 0.0) * out.get("fit_subtype_nb_ll", 0.0)
    return out


def particles_table(exp: Experiment) -> pd.DataFrame:
    """Every parameter set of the experiment, keyed by run id (str): particles.csv rows (p0000, ...) plus
    the runs that recorded their own params (slurm-workflows explorations and searches)."""
    parts = []
    f = exp.out_dir / "particles.csv"
    if f.exists():
        p = pd.read_csv(f)
        parts.append(p.assign(particle=[run_id(int(i)) for i in p.particle]))
    rows = []
    for j in sorted((exp.out_dir / "runs").glob("*.json")):
        meta = json.loads(j.read_text())
        if "params" in meta and not str(meta["particle"]).startswith("p"):
            rows.append({"particle": str(meta["particle"]), **meta["params"]})
    if rows:
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["particle"])


def score(exp: Experiment) -> pd.DataFrame:
    """Score every finished run; rank by fit_score (higher = better)."""
    particles = particles_table(exp)
    obs = observed(exp)
    mix_fit = observed_subtype_mix(exp, exp.fit_start, exp.fit_end)
    mix_proj = observed_subtype_mix(exp, exp.proj_start, exp.proj_end) if exp.proj_start else None
    rows = []
    for i in particles.particle:
        p = run_path(exp, i)
        if not p.exists():
            continue
        r = score_run(exp, pd.read_parquet(p), obs, mix_fit, mix_proj, diagnostics=True)
        rows.append({"particle": i, **r})
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s = s.merge(particles, on="particle").sort_values("fit_score", ascending=False)
    s["rank"] = np.arange(1, len(s) + 1)
    s.to_csv(exp.out_dir / "scores.csv", index=False)
    return s


# --------------------------------------------------------------------------------------
# the projection as an ensemble
# --------------------------------------------------------------------------------------

#: the hub's quantile levels (23)
QUANTILES = np.array([0.01, 0.025, *np.arange(0.05, 0.951, 0.05).round(2), 0.975, 0.99])
#: season-to-season ratio of all-age hospitalization rates, CDC burden estimates 2010-11 .. 2024-25 (pairs
#: touching 2020-21 / 2021-22 dropped; n = 10): geometric mean and sd of the log. p10-p90 0.47-1.95.
INTENSITY_RATIO = (1.10, 0.72)


def intensity_weights(scores: pd.DataFrame, members: list, ratio=INTENSITY_RATIO) -> np.ndarray:
    """Weights for an ensemble's members so the model's next-season / fit-season admissions ratio follows the
    historical season-to-season ratio (log-normal, INTENSITY_RATIO). Each run draws its next-season intensity
    (tau_next_mult) from a wide prior; this sets that draw's distribution from the national record, not from the
    projected season, so the projection stays out of sample. Normalized; equal weights if any ratio is missing."""
    sc = scores.set_index("particle").loc[members]
    r = np.log(sc.proj_model_total.to_numpy() / np.maximum(sc.fit_model_total.to_numpy(), 1e-9))
    if not np.isfinite(r).all():
        return np.full(len(members), 1.0 / len(members))
    z = (r - np.log(ratio[0])) / ratio[1]
    w = np.exp(-0.5 * z ** 2)
    return w / w.sum()


def weighted_quantiles(x: np.ndarray, w: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantiles of each row of x (rows = weeks, columns = members) under member weights w: (len(levels), rows)."""
    o = np.argsort(x, axis=1)
    xs = np.take_along_axis(x, o, axis=1)
    cw = np.cumsum(w[o], axis=1) - 0.5 * w[o]            # mid-point plotting positions
    return np.array([[np.interp(l, cw[i], xs[i]) for i in range(len(x))] for l in levels])


def wis(y: np.ndarray, q: np.ndarray, levels: np.ndarray = QUANTILES) -> np.ndarray:
    """Weighted interval score per column of y (Bracher et al. 2021): q is (len(levels), len(y)), levels
    symmetric around a median. One member (median only) reduces to the absolute error."""
    med = q[np.argmin(np.abs(levels - 0.5))]
    lo = [i for i, l in enumerate(levels) if l < 0.5]
    s = 0.5 * np.abs(y - med)
    for i in lo:
        a = 2 * levels[i]
        l, u = q[i], q[len(levels) - 1 - i]
        s = s + a / 2 * ((u - l) + 2 / a * np.maximum(l - y, 0) + 2 / a * np.maximum(y - u, 0))
    return s / (len(lo) + 0.5)


def ensemble_skill(exp: Experiment, scores: pd.DataFrame, weekly: pd.DataFrame, obs: pd.DataFrame,
                   top: int = 20, window: str = "proj", weights: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Score a window (default the out-of-sample projection) by the ensemble of the `top` runs by fit_score,
    equally weighted, as the hub scores a submission: each run's next-season mix is a prior draw, so the
    single best fit's projection is one sample, not the model's projection.

    Returns (metrics, weekly quantile table). Metrics: WIS (mean over NHSN weeks) and the best run's mean
    absolute error on the same weeks (its WIS as a one-member ensemble); 50% / 90% interval coverage; season
    total (observed, ensemble p10/p50/p90, and the observed total's percentile among the members); peak week
    (observed, ensemble median, median error in weeks); for the projection, the members' mean subtype shares
    and how many drew each subtype absent (< 6% of infections). Model admissions include the background.
    weights="history" weights the members by intensity_weights (projection window only; the fit window is
    unaffected by next-season draws); "ess" in the metrics is the weights' effective sample size."""
    a, b = (exp.fit_start, exp.fit_end) if window == "fit" else (exp.proj_start, exp.proj_end)
    members = scores.particle.head(top).tolist()
    o = obs[obs.age_group == "0-130"].set_index("date").value
    o = o[(o.index >= pd.Timestamp(a)) & (o.index <= pd.Timestamp(b))].dropna()
    wk = weekly[weekly.particle.isin(members) & (weekly.date >= pd.Timestamp(a)) & (weekly.date <= pd.Timestamp(b))]
    m = wk.groupby(["date", "particle"]).admissions.sum().unstack().reindex(columns=members).fillna(0.0)
    m = with_background(exp, window, m.reindex(o.index.union(m.index)).fillna(0.0))
    mo = m.reindex(o.index)
    w = intensity_weights(scores, members) if weights == "history" and window == "proj" \
        else np.full(len(members), 1.0 / len(members))
    q = weighted_quantiles(mo.to_numpy(), w, QUANTILES)
    y = o.to_numpy()
    tot = mo.sum()
    wq = lambda v, l: float(weighted_quantiles(np.asarray(v, dtype=float)[None, :], w, np.array([l]))[0, 0])
    pk = mo.idxmax()
    pk_err = (pk - o.idxmax()).dt.days / 7.0
    out = {"window": window, "weights": weights or "equal", "members": len(members),
           "ess": float(1 / np.sum(w ** 2)), "weeks": len(o),
           "wis": float(wis(y, q).mean()), "best_mae": float(np.abs(y - mo[members[0]].to_numpy()).mean()),
           "cover50": float(np.mean((y >= q[QUANTILES == 0.25][0]) & (y <= q[QUANTILES == 0.75][0]))),
           "cover90": float(np.mean((y >= q[QUANTILES == 0.05][0]) & (y <= q[QUANTILES == 0.95][0]))),
           "obs_total": float(y.sum()), "total_p10": wq(tot, 0.1),
           "total_p50": wq(tot, 0.5), "total_p90": wq(tot, 0.9),
           "obs_total_pct": float(np.sum(w * ((tot < y.sum()) + 0.5 * (tot == y.sum())))),
           "obs_peak_week": o.idxmax().date().isoformat(),
           "peak_week_p50": (pk.min() + (pk - pk.min()).median()).normalize().date().isoformat(),
           "peak_err_wk_p50": wq(pk_err, 0.5)}
    sc = scores.set_index("particle").loc[members]
    for s in SUBTYPES:
        c = f"{window}_model_share_{s}"
        if c in sc:
            out[f"share_{s}"] = float(np.sum(w * sc[c].to_numpy()))
            out[f"absent_{s}"] = int((sc[c] < 0.06).sum())
            out[f"obs_share_{s}"] = float(sc[f"{window}_obs_share_{s}"].iloc[0])
    qt = pd.DataFrame(weighted_quantiles(m.to_numpy(), w, QUANTILES).T, index=m.index,
                      columns=[f"q{l:g}" for l in QUANTILES])
    qt["obs"] = o.reindex(qt.index)
    return out, qt.reset_index()
