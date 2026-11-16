"""In-season forecasting from pools of free-running particles (study kind "forecast"; first study
flu-forecasting-FluTURE-2026-27, docs/forecasting.md in that checkout).

A PARTICLE is one finished restart run (scenario B, 2026-08-09 .. the study's proj_end): a calibration fit's
searched parameters + one draw of the next-season parameters, simulated once. Particles are never re-run per
forecast; data enter only through the rules that weight them, so one pool serves every forecast date.

    S = study()                                      # FLU_ROUND/study.json
    loc = location(S, "VA", ["p00"], through="2026-11-07", as_of="2026-11-16")
    fc = rule_forecast(loc, Rule("aligned", "heldout", 30), loc.T, S["forecast"])
    q = sample_quantiles(fc.F, fc.w, k, n, seed)    # date x quantile level
    make(S, through, as_of, out)                     # every rule + baseline, tests, FluSight files, manifest

A RULE (variant, weights, ess) turns data to a cutoff c into per-particle forecasts F_i(t >= c) and weights w_i:
  variant "raw"      F_i = the particle's trajectory mu_i (model admissions floored at the likelihood background,
                     as the calibration's likelihood sees them)
          "aligned"  F_i = r_i(c) mu_i with r_i(c) = exp(mean over the last align_weeks weeks <= c of
                     log((y + 1) / (mu_i + 1))), clipped to [1/align_clip, align_clip]: the particle's shape and
                     growth from its own dynamics, its level from the data at the origin
  weights "fit"      log w_i ~ NB log-likelihood of mu_i on the weeks to c (fit_weeks, null = all weeks of the
                     season): "fit up to date c"
          "heldout"  the screening cycle: the particle's forecast made at c - test_gap_weeks (same variant) is
                     scored on the test_gap_weeks weeks after it, up to c (NB). For "raw" this is the fit restricted
                     to the last weeks; for "aligned" it measures what the particle gets right beyond the level:
                     its growth over the next weeks
  ess                weights exp(score / T), T set so the effective sample size is ess (proj.tempered_weights)
Forecast samples: `samples` particle draws by systematic resampling on w, each week's value ~ NB(F_i(t), nb_size)
(the calibration's observation model; without it a near-deterministic ensemble is far too sharp for WIS).

THE TEST (decided 2026-11-16: "look at T-2 and evaluate T-1 and T against it"): each rule is applied with the data
to T - test_gap_weeks only (its heldout weights then use T-4 -> T-3, T-2), and its forecast scored by WIS on the
weeks after, through T. The forecast itself applies every rule with the data to T. The flat baseline (FluSight
baseline-like: last value + bootstrapped, symmetrized weekly changes of the location's past NHSN) is scored alike,
so relative WIS is available.

FluSight conventions: reference_date = the Saturday after the last observed week T (the forecast is made in the
week NHSN publishes T), horizon h -> target_end_date = reference_date + 7 h, h = -1 is T itself (revised later);
23 quantile levels; target "wk inc flu hosp"; location = FIPS ("51", "25", "53", "US").
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from . import calib
from .experiment import FLU_ROUND, REPO, load
from .proj import ess as ess_of
from .proj import tempered_weights

QUANTILES = calib.QUANTILES
WEEK = pd.Timedelta(days=7)
BASELINE = "baseline-flat"


# --------------------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------------------

def study_root() -> Path:
    return FLU_ROUND


def study(root: Path | None = None) -> dict:
    f = Path(root or study_root()) / "study.json"
    if not f.exists():
        raise FileNotFoundError(f"{f}: set FLU_ROUND to a forecasting study's checkout (source its env.sh)")
    s = json.loads(f.read_text())
    if s.get("kind") != "forecast":
        raise ValueError(f"{f}: kind {s.get('kind')!r}, not a forecasting study")
    return s


def resolve(p: str, root: Path | None = None) -> Path:
    p = Path(os.path.expandvars(p))
    return p if p.is_absolute() else Path(root or study_root()) / p


def snapshot_dir(as_of: str, root: Path | None = None) -> Path:
    return Path(root or study_root()) / "data" / "snapshots" / "nhsn" / as_of


def latest_snapshot(root: Path | None = None) -> str:
    d = sorted(p.name for p in (Path(root or study_root()) / "data" / "snapshots" / "nhsn").glob("20??-??-??")
               if (p / "time-series.csv").exists())
    if not d:
        raise FileNotFoundError("no NHSN snapshot: fc.py snapshot")
    return d[-1]


def observed(as_of: str, st: str, root: Path | None = None) -> pd.Series:
    """All-age weekly NHSN admissions of one location in snapshot as_of (date-indexed, every reported week)."""
    t = _snapshot(str(snapshot_dir(as_of, root) / "time-series.csv"))
    s = t[(t.location == st) & (t.age_group == "0-130")].set_index("date").observation.sort_index()
    return s.astype(float)


_SNAP: dict[str, pd.DataFrame] = {}


def _snapshot(f: str) -> pd.DataFrame:
    if f not in _SNAP:
        _SNAP[f] = pd.read_csv(f, dtype={"location": str}, parse_dates=["date"])
    return _SNAP[f]


def fips(st: str) -> str:
    import flu_data as fd                     # FLU_ROUND/notebooks is on the path (calib.py)
    inv = {v: k for k, v in fd.FIPS_TO_ABBR.items()}
    return "US" if st == "US" else inv[st]


# --------------------------------------------------------------------------------------
# pools of particles
# --------------------------------------------------------------------------------------

def pool_files(S: dict, st: str, pool: str) -> list[Path]:
    pats = S["pools"][pool]["runs"]
    pats = pats.get(st, pats.get("{ST}", []))
    out = []
    for p in pats:
        out += sorted(Path(f) for f in glob.glob(str(resolve(p.replace("{ST}", st)))))
    return out


def pool_particle_id(pool: str, f: Path) -> str:
    """p00:proj_prodv3_VA/m0001 -> 'p00:prodv3:m0001' (the member id alone repeats across experiments)."""
    exp = f.parent.parent.parent.name.removeprefix("proj_").rsplit("_", 1)[0]
    return f"{pool}:{exp}:{f.stem}"


def load_pool(S: dict, st: str, pools: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Y, meta): Y = weekly all-age model admissions, week-ending date x particle, cut at proj_end; meta = one row
    per particle (pool, file, source fit, next-season parameters)."""
    end = pd.Timestamp(S["calendar"]["proj_end"])
    cols, rows = {}, []
    for pool in pools:
        for f in pool_files(S, st, pool):
            pid = pool_particle_id(pool, f)
            if pid in cols:
                continue
            w = pd.read_parquet(f, columns=["date", "admissions"]).groupby("date").admissions.sum()
            if w.index.max() < end:
                continue                                       # unfinished or a shorter run
            cols[pid] = w[w.index <= end]
            js = f.with_suffix(".json")
            meta = json.loads(js.read_text()) if js.exists() else {}
            rows.append({"particle": pid, "pool": pool, "file": str(f), "source": meta.get("source"),
                         **{f"param_{k}": v for k, v in meta.get("params", {}).items()}})
    if not cols:
        raise RuntimeError(f"{st}: no finished particles in pools {pools}")
    return pd.DataFrame(cols).sort_index(), pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# one location at one cutoff
# --------------------------------------------------------------------------------------

@dataclass
class Location:
    st: str
    T: pd.Timestamp            # the last observed week used (the cutoff)
    mu: pd.DataFrame           # date x particle: model means (floored at the background)
    y: pd.Series               # observed, anchor .. T
    hist: pd.Series            # observed, every week to T (the baseline's history)
    meta: pd.DataFrame
    background: float
    k: float


def location(S: dict, st: str, pools: list[str], through, as_of: str, root: Path | None = None) -> Location:
    fcfg = S["forecast"]
    Y, meta = load_pool(S, st, pools)
    exp = load(f"{S['experiment_family']}_{st}")
    b = calib.background(exp, "proj")
    mu = pd.DataFrame(calib.with_background(exp, "proj", Y.to_numpy(float)), index=Y.index, columns=Y.columns)
    o = observed(as_of, st, root)
    T = pd.Timestamp(through)
    hist = o[o.index <= T].dropna()
    if T not in hist.index:
        raise ValueError(f"{st}: no NHSN value for {T.date()} in snapshot {as_of}")
    y = hist[hist.index >= mu.index.min()]
    return Location(st, T, mu, y, hist, meta, b, float(fcfg.get("nb_size", exp.likelihood["nb_size"])))


def nb_logpmf(y: np.ndarray, mu: np.ndarray, k: float) -> np.ndarray:
    mu = np.maximum(mu, 1e-6)
    return gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu))


@dataclass(frozen=True)
class Rule:
    variant: str               # raw | aligned
    weights: str               # fit | heldout
    ess: float

    @property
    def name(self) -> str:
        return f"{self.variant}-{self.weights}-ess{self.ess:g}"

    @staticmethod
    def parse(name: str) -> "Rule":
        v, w, e = name.split("-")
        return Rule(v, w, float(e.removeprefix("ess")))


def rules(fcfg: dict) -> list[Rule]:
    return [Rule(v, w, float(e)) for v in fcfg["variants"] for w in fcfg["weights"] for e in fcfg["ess"]]


def primary(fcfg: dict) -> Rule:
    p = fcfg["primary"]
    return Rule(p["variant"], p["weights"], float(p["ess"]))


def align_ratio(loc: Location, c: pd.Timestamp, fcfg: dict) -> np.ndarray:
    L = int(fcfg.get("align_weeks", 2))
    yy = loc.y[(loc.y.index <= c)].tail(L)
    m = loc.mu.reindex(yy.index).to_numpy()
    r = np.exp(np.mean(np.log((yy.to_numpy()[:, None] + 1.0) / (m + 1.0)), axis=0))
    clip = float(fcfg.get("align_clip", 4.0))
    return np.clip(r, 1 / clip, clip)


def particle_forecast(loc: Location, variant: str, c: pd.Timestamp, fcfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """F (dates >= c x particles) for an origin c, and the alignment ratio (1 for raw)."""
    F = loc.mu[loc.mu.index >= c]
    if variant == "raw":
        return F, np.ones(F.shape[1])
    if variant == "aligned":
        r = align_ratio(loc, c, fcfg)
        return F * r, r
    raise ValueError(f"variant {variant!r}: raw | aligned")


def fit_ll(loc: Location, c: pd.Timestamp, fcfg: dict) -> np.ndarray:
    yy = loc.y[loc.y.index <= c]
    if fcfg.get("fit_weeks"):
        yy = yy.tail(int(fcfg["fit_weeks"]))
    return nb_logpmf(yy.to_numpy()[:, None], loc.mu.reindex(yy.index).to_numpy(), loc.k).sum(axis=0)


def heldout_ll(loc: Location, variant: str, c: pd.Timestamp, fcfg: dict) -> np.ndarray:
    """Each particle's forecast from c - gap, scored on the gap weeks after it, through c."""
    c0 = c - int(fcfg.get("test_gap_weeks", 2)) * WEEK
    F, _ = particle_forecast(loc, variant, c0, fcfg)
    yy = loc.y[(loc.y.index > c0) & (loc.y.index <= c)]
    return nb_logpmf(yy.to_numpy()[:, None], F.reindex(yy.index).to_numpy(), loc.k).sum(axis=0)


@dataclass
class RuleForecast:
    rule: Rule
    c: pd.Timestamp
    F: pd.DataFrame
    w: np.ndarray
    score: np.ndarray
    ratio: np.ndarray
    temperature: float

    @property
    def ess(self) -> float:
        return ess_of(self.w)


def rule_forecast(loc: Location, rule: Rule, c, fcfg: dict) -> RuleForecast:
    """The rule applied with the data to c (nothing after c is read)."""
    c = pd.Timestamp(c)
    score = fit_ll(loc, c, fcfg) if rule.weights == "fit" else heldout_ll(loc, rule.variant, c, fcfg) \
        if rule.weights == "heldout" else None
    if score is None:
        raise ValueError(f"weights {rule.weights!r}: fit | heldout")
    w, t = tempered_weights(score, rule.ess)
    F, r = particle_forecast(loc, rule.variant, c, fcfg)
    return RuleForecast(rule, c, F, w, score, r, t)


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def sample_quantiles(F: pd.DataFrame, w: np.ndarray, k: float, n: int, seed: int,
                     levels: np.ndarray = QUANTILES) -> pd.DataFrame:
    """date x level: n draws, particle by systematic resampling on w, value ~ NB(F_i(t), k)."""
    rng = np.random.default_rng(seed)
    u = (rng.random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), len(w) - 1)
    m = np.maximum(F.to_numpy()[:, idx], 1e-6)
    x = rng.negative_binomial(k, k / (k + m))
    return pd.DataFrame(np.quantile(x, levels, axis=1).T, index=F.index, columns=levels)


def baseline_quantiles(hist: pd.Series, c, dates, n: int, seed: int, levels: np.ndarray = QUANTILES) -> pd.DataFrame:
    """Flat baseline (FluSight-baseline-like): median = the value at c; spread = sums of bootstrapped weekly
    changes of the location's history to c, symmetrized; floored at 0."""
    c = pd.Timestamp(c)
    h = hist[hist.index <= c]
    d = h.diff().dropna().to_numpy()
    d = np.concatenate([d, -d])
    rng = np.random.default_rng(seed)
    out = {}
    for t in dates:
        steps = int(round((pd.Timestamp(t) - c) / WEEK))
        x = h.iloc[-1] + (rng.choice(d, size=(n, steps)).sum(axis=1) if steps > 0 else np.zeros(n))
        out[pd.Timestamp(t)] = np.quantile(np.maximum(x, 0), levels)
    return pd.DataFrame(out, index=levels).T


def wis_of(q: pd.DataFrame, y: pd.Series) -> pd.Series:
    """WIS per date (q: date x level) against y (date-indexed); dates without y are dropped."""
    yy = y.reindex(q.index).dropna()
    if yy.empty:
        return pd.Series(dtype=float)
    qq = q.loc[yy.index].to_numpy().T
    return pd.Series(calib.wis(yy.to_numpy(), qq, np.asarray(q.columns, float)), index=yy.index)


def flusight_rows(q: pd.DataFrame, ref: pd.Timestamp, st: str, target: str) -> pd.DataFrame:
    long = q.rename_axis("target_end_date").reset_index().melt("target_end_date", var_name="level", value_name="value")
    long["horizon"] = ((long.target_end_date - ref) / WEEK).round().astype(int)
    long["output_type_id"] = long.level.map(lambda l: f"{float(l):.3f}".rstrip("0").rstrip("."))
    return long.assign(reference_date=ref, target=target, location=fips(st), output_type="quantile")[
        ["reference_date", "target", "horizon", "target_end_date", "location", "output_type", "output_type_id",
         "value"]].sort_values(["horizon", "value"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# a forecast (live or backtest): every rule + baseline, the test, files, manifest
# --------------------------------------------------------------------------------------

def _git(repo: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"repo": str(repo), "commit": run("rev-parse", "--short", "HEAD"), "branch": run("branch", "--show-current"),
            "dirty_files": len([x for x in run("status", "--porcelain", "--untracked-files=no").splitlines() if x])}


def reference_date(through) -> pd.Timestamp:
    return pd.Timestamp(through) + WEEK


def make(S: dict, through, as_of: str, out: Path, states: list[str] | None = None, pools: list[str] | None = None,
         kind: str = "live", root: Path | None = None, verbose: bool = True) -> Path:
    """One forecast with the data to `through` from snapshot `as_of` -> out/:
      manifest.json                    calendar, snapshot, pools, rules, primary, code versions, test summary
      <ref>-<model_id>.csv             FluSight file of the primary rule (horizons -1..3, all locations)
      rules/<rule>.csv                 every rule and the baseline, all horizons to proj_end (FluSight columns)
      tests.csv                        the screening test: each rule applied at T - gap, WIS on the weeks to T
      tests/<rule>.csv                 the test forecasts (FluSight columns, reference_date = T - gap + 7)
      <ST>/particles.csv               per particle: weights and scores of every rule (forecast and test), ratios
    """
    fcfg = S["forecast"]
    states = states or S["locations"]
    pools = pools or fcfg["pools"]
    T = pd.Timestamp(through)
    ref = reference_date(T)
    gap = int(fcfg.get("test_gap_weeks", 2))
    c_test = T - gap * WEEK
    n, target = int(fcfg.get("samples", 4000)), fcfg.get("target", "wk inc flu hosp")
    all_rules = rules(fcfg)
    prim = primary(fcfg)
    if prim not in all_rules:
        all_rules.append(prim)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules").mkdir(exist_ok=True)
    (out / "tests").mkdir(exist_ok=True)
    rows_fc: dict[str, list] = {r: [] for r in [*(x.name for x in all_rules), BASELINE]}
    rows_test: dict[str, list] = {r: [] for r in rows_fc}
    tests, loc_info = [], {}
    for st in states:
        loc = location(S, st, pools, T, as_of, root)
        yT = loc.y
        part = loc.meta[["particle", "pool", "source"]].copy()
        test_dates = [d for d in loc.mu.index if c_test < d <= T]
        fc_dates = list(loc.mu.index[loc.mu.index >= T])
        for rule in all_rules:
            for tag, c, dates, rows in (("test", c_test, test_dates, rows_test), ("fc", T, fc_dates, rows_fc)):
                rf = rule_forecast(loc, rule, c, fcfg)
                F = rf.F.loc[[d for d in rf.F.index if d in set(dates) or tag == "fc"]]
                q = sample_quantiles(F, rf.w, loc.k, n, _seed(ref.date(), st, rule.name, tag))
                rows[rule.name].append(flusight_rows(q, c + WEEK, st, target))
                part[f"{tag}_w_{rule.name}"] = rf.w
                if rule.ess == prim.ess:                       # scores do not depend on ess: keep one copy
                    part[f"{tag}_score_{rule.variant}-{rule.weights}"] = rf.score
                if rule.variant == "aligned" and rule == prim:
                    part[f"{tag}_ratio"] = rf.ratio
                if tag == "test":
                    wv = wis_of(q, yT)
                    tests += [{"location": st, "rule": rule.name, "variant": rule.variant, "weights": rule.weights,
                               "ess_target": rule.ess, "ess": rf.ess, "temperature": rf.temperature,
                               "target_end_date": d, "horizon": int(round((d - (c + WEEK)) / WEEK)),
                               "observed": float(yT[d]), "median": float(q.loc[d, 0.5]), "wis": float(v)}
                              for d, v in wv.items()]
        for tag, c, dates, rows in (("test", c_test, test_dates, rows_test), ("fc", T, fc_dates, rows_fc)):
            qb = baseline_quantiles(loc.hist, c, dates, n, _seed(ref.date(), st, BASELINE, tag))
            rows[BASELINE].append(flusight_rows(qb, c + WEEK, st, target))
            if tag == "test":
                tests += [{"location": st, "rule": BASELINE, "variant": "baseline", "weights": "", "ess_target": np.nan,
                           "ess": np.nan, "temperature": np.nan, "target_end_date": d,
                           "horizon": int(round((d - (c + WEEK)) / WEEK)), "observed": float(yT[d]),
                           "median": float(qb.loc[d, 0.5]), "wis": float(v)} for d, v in wis_of(qb, yT).items()]
        (out / st).mkdir(exist_ok=True)
        part.to_csv(out / st / "particles.csv", index=False)
        loc_info[st] = {"particles": int(loc.mu.shape[1]), "by_pool": part.pool.value_counts().to_dict(),
                        "last_week": f"{T.date()}", "last_value": float(loc.y.iloc[-1]), "background": loc.background}
        if verbose:
            print(f"  {st}: {loc.mu.shape[1]} particles, NHSN {T.date()} = {loc.y.iloc[-1]:.0f}", flush=True)
    for name, parts in rows_fc.items():
        pd.concat(parts).to_csv(out / "rules" / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    for name, parts in rows_test.items():
        pd.concat(parts).to_csv(out / "tests" / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    tt = pd.DataFrame(tests)
    tt.to_csv(out / "tests.csv", index=False, date_format="%Y-%m-%d")
    hz = fcfg.get("horizons", [-1, 0, 1, 2, 3])
    p = pd.concat(rows_fc[prim.name])
    p[p.horizon.isin(hz)].to_csv(out / f"{ref.date()}-{fcfg.get('model_id', 'UVA-escape_flu')}.csv", index=False,
                                 date_format="%Y-%m-%d")
    summ = (tt.groupby("rule").wis.mean().sort_values()).round(2).to_dict() if len(tt) else {}
    snap = snapshot_dir(as_of, root) / "meta.json"
    manifest = {
        "study": S["name"], "kind": kind, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_date": f"{ref.date()}", "data_through": f"{T.date()}", "snapshot": as_of,
        "snapshot_meta": json.loads(snap.read_text()) if snap.exists() else None,
        "test": {"origin": f"{c_test.date()}", "scored_weeks": [f"{d.date()}" for d in
                                                                 sorted(set(tt.target_end_date))] if len(tt) else [],
                 "mean_wis_by_rule": summ},
        "pools": pools, "locations": loc_info, "primary": prim.name, "rules": [r.name for r in all_rules],
        "settings": {k: v for k, v in fcfg.items() if not k.startswith("_")},
        "scenario": S.get("scenario"), "proj_end": S["calendar"]["proj_end"],
        "code": {"escape-flu-smh": _git(REPO), "pipeline": _git(Path(root or study_root()))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return out


# --------------------------------------------------------------------------------------
# scoring issued forecasts against newer data
# --------------------------------------------------------------------------------------

def score(dirs: list[Path], truth_as_of: str, root: Path | None = None) -> pd.DataFrame:
    """WIS, absolute error of the median and 50/90% coverage of every rule's forecast in each directory against
    snapshot truth_as_of (rows only where truth exists; h = -1 included: the forecast's last week, revised)."""
    import flu_data as fd
    out = []
    for d in dirs:
        m = json.loads((d / "manifest.json").read_text())
        for f in sorted((d / "rules").glob("*.csv")):
            q = pd.read_csv(f, dtype={"location": str, "output_type_id": str},
                            parse_dates=["reference_date", "target_end_date"])
            for loc_code, g in q.groupby("location"):
                st = fd.FIPS_TO_ABBR.get(loc_code, loc_code)
                y = observed(truth_as_of, st, root)
                piv = g.pivot_table(index="target_end_date", columns="output_type_id", values="value")
                piv.columns = piv.columns.astype(float)
                piv = piv[sorted(piv.columns)]
                wv = wis_of(piv, y)
                for t, v in wv.items():
                    r = piv.loc[t]
                    out.append({"kind": m["kind"], "reference_date": m["reference_date"], "data_through": m["data_through"],
                                "snapshot": m["snapshot"], "truth": truth_as_of, "rule": f.stem, "location": st,
                                "target_end_date": t.date().isoformat(),
                                "horizon": int(round((t - pd.Timestamp(m["reference_date"])) / WEEK)),
                                "observed": float(y[t]), "median": float(r[0.5]), "wis": float(v),
                                "ae": abs(float(r[0.5]) - float(y[t])),
                                "cov50": bool(r[0.25] <= y[t] <= r[0.75]), "cov90": bool(r[0.05] <= y[t] <= r[0.95])})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# the pool's loose pre-screen (screen runs -> restart/selected.csv)
# --------------------------------------------------------------------------------------

def prescreen(name: str, as_of: str, n_keep: int = 384, band: float = 4.0, weeks: int = 3, seed: int | None = None,
              root: Path | None = None) -> pd.DataFrame:
    """Screen runs (proj_<name>/screen/<sid>/, cut at production.screen_end) -> the members to run in full.
    Loose on purpose (the forecasts' rules do the selecting, and every later forecast date reuses the pool):
      1. level: the run's geometric-mean ratio (mu + 1) / (y + 1) over the last `weeks` NHSN weeks to
         production.data_end within [1/band, band]
      2. if more than n_keep pass: n_keep by systematic PPS (proj.pps_sample) on NB fit weights over the season to
         data_end, tempered to ESS n_keep (mild: keeps the spread, thins the worst)
    Writes proj_<name>/restart/selected.csv (the members' rows, ids unchanged) and restart/prescreen.csv."""
    from . import proj
    exp = load(name)
    sid = proj.scenario_ids(exp)[0]
    data_end = pd.Timestamp(exp.production["data_end"])
    st = exp.state
    y = observed(as_of, st, root)
    y = y[(y.index >= pd.Timestamp(exp.production["restart_anchor"])) & (y.index <= data_end)].dropna()
    cols = {}
    for f in sorted((proj.out_dir(exp) / "screen" / sid).glob("*.parquet")):
        w = pd.read_parquet(f, columns=["date", "admissions"]).groupby("date").admissions.sum()
        cols[f.stem] = w
    if not cols:
        raise RuntimeError(f"{name}: no screen runs in {proj.out_dir(exp) / 'screen' / sid}")
    M = pd.DataFrame(cols).reindex(y.index)
    if M.isna().any().any():
        raise RuntimeError(f"{name}: screen runs end before {data_end.date()} (production.screen_end)")
    mu = pd.DataFrame(calib.with_background(exp, "proj", M.to_numpy(float)), index=M.index, columns=M.columns)
    last = y.tail(weeks)
    ratio = np.exp(np.mean(np.log((mu.loc[last.index].to_numpy() + 1) / (last.to_numpy()[:, None] + 1)), axis=0))
    ll = nb_logpmf(y.to_numpy()[:, None], mu.to_numpy(), float(exp.likelihood["nb_size"])).sum(axis=0)
    tab = pd.DataFrame({"member": mu.columns, "ratio": ratio, "fit_ll": ll})
    tab["level_ok"] = (tab.ratio >= 1 / band) & (tab.ratio <= band)
    ok = tab[tab.level_ok].reset_index(drop=True)
    rng = np.random.default_rng(seed if seed is not None else exp.seed)
    if len(ok) > n_keep:
        w, t = tempered_weights(ok.fit_ll.to_numpy(), n_keep)
        pick, incl = proj.pps_sample(w, n_keep, rng)
        ok["inclusion"] = incl
        keep = set(ok.member.iloc[pick])
    else:
        ok["inclusion"] = 1.0
        keep = set(ok.member)
    tab = tab.merge(ok[["member", "inclusion"]], on="member", how="left").fillna({"inclusion": 0.0})
    tab["kept"] = tab.member.isin(keep)
    mem = proj.load_members(exp)
    d = proj.out_dir(exp) / "restart"
    d.mkdir(parents=True, exist_ok=True)
    mem[mem.member.isin(keep)].to_csv(d / "selected.csv", index=False)
    tab.assign(as_of=as_of, data_end=f"{data_end.date()}", band=band, weeks=weeks).to_csv(d / "prescreen.csv",
                                                                                           index=False)
    print(f"{name}: {len(tab)} screened, {int(tab.level_ok.sum())} within x{band:g} of NHSN over the last {weeks} weeks "
          f"to {data_end.date()}, {len(keep)} kept -> {d / 'selected.csv'}", flush=True)
    return tab
