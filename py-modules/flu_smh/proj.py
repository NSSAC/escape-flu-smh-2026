"""Hub projections from a calibrated production experiment: paired trajectories under every scenario.

    members(exp, n=300, top=20)        -> proj_<name>/members.csv: member, run_grouping, source run, the
                                          searched parameters of a top calibration run + the member's own draws
    scenario_experiment(exp, sid, mode) the Experiment one scenario runs under (its own build dir)
    prepare(exp, sid, mode)            schedule -> build -> input.h5 (shared by the scenarios of a mode)
    run_member(exp, sid, member, mode) -> proj_<name>/<mode>/<sid>/<member>.parquet: weekly infections /
                                          admissions / deaths by subtype x age; .json: configs, S0 at the origin
    select(exp, mode, n, weights)      optional: after an oversampled scenario-B pass, keep n members
                                          (production 2026-27: weights="early" on a screen pass, then
                                          weights="guard" on the full runs, against the state's past seasons)
    collect(exp, mode)                 every finished trajectory, long, plus the S0 table

MODES (restart is the production choice, 2026-10-27: ~0.35x the cost of continuous, and a pair shares its
history exactly)
  continuous  the calibration's own run from its anchor (2025-08-03) with the 2026-27 campaign replaced by the
              scenario's: exactly the procedure the backtests (bt2023k, bt2024k) score. ~672 ticks a run, and each
              scenario re-simulates 2025-26 with its own noise (the simulator has no seed).
  restart     one season from production["restart_anchor"] (2026-08-09) on production["restart_person_file"]
              (the NHSN-reconstructed history up to that day, flu_ic): the next-season block applies from tick 0
              and the absolute-tick configs move with the anchor. ~300 ticks a run (~0.36x the cost of the
              two-season run), and paired runs share their history exactly. ltimm_frac_<sub> are re-solved (per
              subtype) so tick-0 S0
              matches production["restart_s0"]: "source" (default: the member's own calibration run's S0 on the
              origin, 2026-10-04, so members keep the calibration's spread of immunity), "calibration" (the median
              of the top runs) or a number. The origin, not the anchor: the calibration runs' R state drains over
              Aug-Sep, and a restart's S0 barely moves before October (~ -0.006 from B's early doses).

MEMBERS AND PAIRING
  Member k (k = 0..n-1) takes the searched parameters of calibration run (k mod top) of the `top` best by
  fit_score (weights="uniform"), or runs drawn with probability ~ exp(fit_score / temperature) (weights="fit"),
  plus its own next-season draws (calib.draws_for with id m<k>): 300 members over 20 runs = 15 next seasons
  each. run_grouping = k + 1 in every scenario, stochastic_run = 1. The hub's S0 tag must match across a pair,
  so the submission takes it from scenario B (smh_submission.py).

S0 (hub definition: the population mean of relative susceptibility on the origin day)
  S0_<sub> = susceptible_mass_<sub> / N at the origin tick; S0 = sum_k w_k S0_k with w = the member's share of
  each subtype in its projected-season infections (S0_A the same over the two A subtypes).
"""
from __future__ import annotations

import dataclasses
import json
import time
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import calib, sim
from .experiment import FLU_ROUND, SUBTYPES, Experiment, load

MODES = ("restart", "continuous", "screen")
#: modes that restart from production["restart_anchor"] (screen = restart cut at the origin)
RESTARTS = ("restart", "screen")
#: member-table columns that are not parameters (parent: the screened member a selected one copies; topup_round:
#: the top-up that drew it, 0 for the first screen selection)
MEMBER_META = ("member", "run_grouping", "source", "source_rank", "source_fit_score", "parent", "topup_round")
#: the guard (select weights="guard"): the state's NHSN seasons since mandatory reporting; a member's weight is 1
#: up to GUARD_FULL x the largest past season and tapers to 0 at GUARD_MAX x it, the lower of its total's and its
#: peak's weights. A season missing weeks (< GUARD_WINTER_WEEKS of its Nov 15 .. Mar 15 weeks: WY, DC, DE) still
#: counts, as a lower bound. GUARD_FULL was 1.0 with the two weights multiplied until 2026-10-30: in preview-v3 VA
#: kept 2 of the 20 members whose peak was 1-2x the record (median kept peak 0.34x), more filtering than intended
GUARD_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
GUARD_WINTER_WEEKS = 15
GUARD_FULL = 1.5
GUARD_MAX = 2.0
#: configs holding absolute ticks (shifted with the anchor in restart mode)
ABS_TICKS = ["seas_peak_tick", "seas_peak_tick_next", *[f"seed_start_{s}" for s in SUBTYPES],
             *[f"seed_start_{s}_next" for s in SUBTYPES]]


def out_dir(exp: Experiment) -> Path:
    return FLU_ROUND / f"proj_{exp.name}"


def scenario_ids(exp: Experiment) -> list[str]:
    return list(exp.production["scenarios"])


def scenario_id(exp: Experiment, s: str) -> str:
    """'B' or 'B-2026-09-29' -> 'B-2026-09-29'."""
    m = [k for k in scenario_ids(exp) if k == s or k.split("-")[0] == s]
    if len(m) != 1:
        raise KeyError(f"{exp.name}: no scenario {s!r} in {scenario_ids(exp)}")
    return m[0]


# --------------------------------------------------------------------------------------
# experiments per scenario and mode
# --------------------------------------------------------------------------------------

def scenario_experiment(exp: Experiment, sid: str, mode: str = "restart") -> Experiment:
    prod = exp.production
    camps = [dict(c, coverage=prod["scenarios"][sid]) if c["season"] == prod["season"] else dict(c)
             for c in exp.campaigns]
    kw = dict(name=f"{exp.name}-{mode}-{sid.split('-')[0]}", campaigns=camps)
    if mode in RESTARTS:
        end = prod.get("end", exp.end) if mode == "restart" else \
            str((pd.Timestamp(prod["origin"]) + pd.Timedelta(days=7)).date())
        kw |= dict(anchor=prod["restart_anchor"], person_file=prod["restart_person_file"],
                   season_break=prod["restart_anchor"], end=end)
    elif mode != "continuous":
        raise ValueError(f"mode must be one of {MODES}")
    return dataclasses.replace(exp, **kw)


def tick_shift(exp: Experiment, mode: str) -> int:
    return exp.tick(exp.production["restart_anchor"]) if mode in RESTARTS else 0


def input_home(exp: Experiment, mode: str) -> Path:
    """The input.h5 the scenarios of a mode share: the calibration's (continuous) or one built for restart."""
    if mode == "continuous":
        return sim.input_path(exp)
    return exp.work_dir.parent / f"{exp.name}-restart-input" / "input.h5"


def prepare(exp: Experiment, sid: str, mode: str = "restart") -> Path:
    """Schedule, render and build the scenario's simulator, on the mode's shared input.h5 (linked; the first
    restart scenario prepared builds it, so prepare one scenario per experiment before the others)."""
    sx = scenario_experiment(exp, sid, mode)
    sx.work_dir.mkdir(parents=True, exist_ok=True)
    home, link = input_home(exp, mode), sim.input_path(sx)
    if home.exists() and not link.exists():
        link.symlink_to(home)
    sim.prepare(sx)                          # builds input.h5 only if there is none
    if not home.exists():
        home.parent.mkdir(parents=True, exist_ok=True)
        link.replace(home)
        link.symlink_to(home)
    return sx.work_dir


# --------------------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------------------

def member_id(k: int) -> str:
    return f"m{k:04d}"


def next_space(exp: Experiment) -> dict:
    """The next-season parameters a member draws: production["next"] (two-stage experiments) or exp.draws."""
    return {k: calib._spec(v) for k, v in (exp.production.get("next") or exp.draws).items()}


def proj_window(exp: Experiment) -> tuple[str, str]:
    w = exp.production.get("proj_window")
    return tuple(w) if w else (exp.proj_start, exp.proj_end)


def early_window(exp: Experiment) -> tuple[str, str]:
    """The projected season's weeks before the origin that the hub lets a model see: restart anchor .. the
    fitting cutoff (production["data_end"]; a backtest: the day before its origin)."""
    p = exp.production
    end = p.get("data_end") or str((pd.Timestamp(p["origin"]) - pd.Timedelta(days=1)).date())
    return p["restart_anchor"], end


def members(exp: Experiment, n: int = 300, top: int = 20, weights: str = "uniform",
            temperature: float = 1.0, seed: int | None = None, design: str = "random",
            per_fit: int | None = None) -> pd.DataFrame:
    """The members' parameter sets (searched values from the calibration + next-season values) -> members.csv.

    design="random"  member k = calibration run (k mod top) (or fit-weighted, weights="fit") + independent draws
                     from next_space (seeded by the experiment seed and the member id): production members.
    design="sobol"   round 2 of the two-stage design: each of the `top` fits x a scrambled Sobol design of
                     `per_fit` points (power of 2) over next_space, the same design for every fit, so a parameter's
                     effect on the projection is seen within each fit. n = top x per_fit."""
    sc = pd.read_csv(exp.out_dir / "scores.csv").sort_values("fit_score", ascending=False)
    searched = list(calib.space(exp))
    best = sc.head(top).reset_index(drop=True)
    sp = next_space(exp)
    rng = np.random.default_rng(seed if seed is not None else exp.seed)
    if design == "sobol":
        per_fit = per_fit or 32
        u = qmc.Sobol(d=len(sp), scramble=True, seed=rng).random(per_fit)
        src = np.repeat(np.arange(len(best)), per_fit)
        nxt = [{k: float(calib.from_unit(spec, u[j % per_fit, i])) for i, (k, spec) in enumerate(sp.items())}
               for j in range(len(src))]
    elif design == "random":
        if weights == "uniform":
            src = np.arange(n) % len(best)
        elif weights == "fit":
            w = np.exp((best.fit_score - best.fit_score.max()) / temperature)
            src = rng.choice(len(best), size=n, p=w / w.sum())
        else:
            raise ValueError("weights: uniform | fit")
        nxt = []
        for k in range(len(src)):
            r = np.random.default_rng([int(exp.seed), zlib.crc32(member_id(k).encode())])
            nxt.append({p: float(calib.from_unit(spec, r.random())) for p, spec in sp.items()})
    else:
        raise ValueError("design: random | sobol")
    rows = []
    for k in range(len(src)):
        b = best.iloc[int(src[k])]
        rows.append({"member": member_id(k), "run_grouping": k + 1, "source": b.particle,
                     "source_rank": int(src[k]) + 1, "source_fit_score": float(b.fit_score),
                     **{p: float(b[p]) for p in searched}, **nxt[k]})
    m = pd.DataFrame(rows)
    d = out_dir(exp)
    d.mkdir(parents=True, exist_ok=True)
    m.to_csv(d / "members.csv", index=False)
    return m


def load_members(exp: Experiment, selected: bool = False, mode: str = "restart") -> pd.DataFrame:
    f = out_dir(exp) / mode / "selected.csv" if selected else out_dir(exp) / "members.csv"
    return pd.read_csv(f)


# --------------------------------------------------------------------------------------
# configs and runs
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def restart_ltimm(name: str, source: str | None = None, escape: tuple = (0.0, 0.0, 0.0)) -> dict:
    """ltimm_frac_<sub> for restart mode: each subtype's tick-0 S0 = the target of production["restart_s0"], per
    subtype, so a member starts with its calibration run's immunity by subtype (a number targets all three)."""
    import informed_priors as ip
    exp = load(name)
    sx = scenario_experiment(exp, scenario_ids(exp)[0], "restart")
    hl = float(exp.fixed.get("infection_half_life_days", 365.0))
    cfg = {**exp.fixed, "infection_waning_per_day": 0.5 ** (1 / hl)}
    s, adult = _s0_history(name, sx.person_path, json.dumps(cfg, sort_keys=True))
    wgt = ip.LTIMM["ltimm_protection"] * (adult + ip.LTIMM["ltimm_child_rel"] * (1 - adult))
    target = exp.production.get("restart_s0", "source")
    # the calibration's S0 on the ORIGIN, not on the restart anchor: on 2026-08-09 the continuous runs still hold
    # the late 2025-26 infections in R (fully protected), which return to S (partly protected) by October, so the
    # anchor's S0 is ~0.06 too low for a restart that starts everyone in S (WY smoke test, 2026-10-27). Per
    # subtype, because one fraction for all three matched the mean but not the mix (a B-heavy simulated 2025-26
    # left B at 0.60 and H1 at 0.69; the reconstruction-based history put all three at ~0.65).
    tick = exp.tick(exp.production["origin"])
    if target == "source" and source is not None:
        target = calibration_s0(exp, tick, runs=[source])
    elif target in ("source", "calibration"):
        target = calibration_s0(exp, tick)
    else:
        target = {k: float(target) for k in SUBTYPES}
    # escape_next_<sub>: a drift jump at the season break removes that share of every agent's protection, so the
    # protected share 1 - S0 shrinks by (1 - e); clipped where the history alone cannot go that high
    target = {k: 1 - (1 - target[k]) * (1 - e) for k, e in zip(SUBTYPES, escape)}
    return {k: float(np.clip((1 - target[k] / s[k]) / wgt, 0.0, 1.0)) for k in SUBTYPES}


@lru_cache(maxsize=None)
def _s0_history(name: str, person_path: Path, cfg_json: str):
    import informed_priors as ip
    return ip.s0_history(person_path, json.loads(cfg_json))


def calibration_s0(exp: Experiment, tick: int, top: int = 20, runs: list[str] | None = None) -> dict:
    """{subtype: median over calibration runs (`runs`, default the top `top` by fit_score) of S0 at `tick`}."""
    if runs is None:
        sc = pd.read_csv(exp.out_dir / "scores.csv").sort_values("fit_score", ascending=False).head(top)
        runs = list(sc.particle)
    n = sim.population(exp)
    vals = []
    for rid in runs:
        f = calib.run_path(exp, rid).with_suffix(".stats.parquet")
        if f.exists():
            st = pd.read_parquet(f)
            r = st[st.tick == tick]
            if len(r):
                vals.append({k: r[f"susceptible_mass_{k}"].iloc[0] / n for k in SUBTYPES})
    if not vals:
        raise RuntimeError(f"{exp.name}: no calibration stats at tick {tick} for {runs[:3]}; set production.restart_s0")
    return {k: float(np.median([v[k] for v in vals])) for k in SUBTYPES}


def member_configs(exp: Experiment, params: dict, mode: str, source: str | None = None) -> dict:
    cfg = calib.configs_for(exp, params)
    if mode not in RESTARTS and any(cfg.get(f"escape_next_{k}", 0.0) for k in SUBTYPES):
        raise ValueError("escape_next_* needs restart mode (it adjusts the restart's S0 target)")
    if mode in RESTARTS:
        sh = tick_shift(exp, mode)
        for k in ABS_TICKS:
            if k in cfg:
                cfg[k] = cfg[k] - sh
        cfg["season_break_tick"] = 0.0
        esc = tuple(float(cfg.pop(f"escape_next_{k}", 0.0)) for k in SUBTYPES)
        for s, f in restart_ltimm(exp.name, source, esc).items():
            cfg[f"ltimm_frac_{s}"] = f
    return cfg


def member_path(exp: Experiment, mode: str, sid: str, mid: str) -> Path:
    return out_dir(exp) / mode / sid / f"{mid}.parquet"


def s0_at(sx: Experiment, stats: pd.DataFrame, weekly: pd.DataFrame, origin: str) -> dict:
    n = sim.population(sx)
    r = stats[stats.tick == sx.tick(origin)].iloc[0]
    s0 = {s: float(r[f"susceptible_mass_{s}"]) / n for s in SUBTYPES}
    inf = weekly[weekly.date >= pd.Timestamp(origin)].groupby("subtype").infections.sum().reindex(SUBTYPES)
    w = (inf / inf.sum()).fillna(1 / 3) if inf.sum() > 0 else pd.Series(1 / 3, index=SUBTYPES)
    wa = w[["A_H1N1", "A_H3N2"]]
    wa = wa / wa.sum() if wa.sum() > 0 else pd.Series(0.5, index=wa.index)
    return {"S0": float(sum(w[s] * s0[s] for s in SUBTYPES)),
            "S0_A": float(wa["A_H1N1"] * s0["A_H1N1"] + wa["A_H3N2"] * s0["A_H3N2"]),
            "S0_AH1": s0["A_H1N1"], "S0_AH3": s0["A_H3N2"], "S0_B": s0["B"],
            **{f"share_{s}": float(w[s]) for s in SUBTYPES}}


def run_member(exp: Experiment | str, sid: str, member: dict, mode: str = "restart",
               verbose: bool = False) -> Path:
    """One trajectory: simulate, extract the weekly table and S0, delete the HDF5."""
    exp = load(exp) if isinstance(exp, str) else exp
    sid = scenario_id(exp, sid)
    sx = scenario_experiment(exp, sid, mode)
    mid = member["member"]
    params = {k: v for k, v in member.items() if k not in MEMBER_META}
    cfg = member_configs(exp, params, mode, member.get("source"))
    h5 = sx.work_dir / "runs" / f"{mid}.h5"
    t0 = time.time()
    sim.run(sx, cfg, h5, verbose=verbose)
    t_run = time.time() - t0
    w = sim.extract(sx, h5, _band(sx))
    st = sim.stats(sx, h5)
    out = member_path(exp, mode, sid, mid)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.assign(member=mid).to_parquet(out, index=False)
    s0 = s0_at(sx, st, w, exp.production["origin"])
    out.with_suffix(".json").write_text(json.dumps(
        {"member": mid, "run_grouping": int(member["run_grouping"]), "scenario_id": sid, "mode": mode,
         "source": member.get("source"), "params": {k: float(v) for k, v in params.items()}, "configs": cfg,
         "s0": s0, "run_seconds": round(t_run, 1), "extract_seconds": round(time.time() - t0 - t_run, 1)},
        indent=1))
    h5.unlink(missing_ok=True)
    return out


_BANDS: dict[Path, np.ndarray] = {}


def _band(sx: Experiment) -> np.ndarray:
    """Node age bands, cached per person file for the life of a worker."""
    if sx.person_path not in _BANDS:
        _BANDS[sx.person_path] = sim.node_age_band(sx)
    return _BANDS[sx.person_path]


def missing(exp: Experiment, mode: str, sids: list[str], mem: pd.DataFrame) -> list[tuple[str, dict]]:
    return [(sid, r) for sid in sids for r in mem.to_dict("records")
            if not member_path(exp, mode, sid, r["member"]).with_suffix(".json").exists()]


# --------------------------------------------------------------------------------------
# selection and collection
# --------------------------------------------------------------------------------------

def member_totals(exp: Experiment, mode: str, sid: str) -> pd.DataFrame:
    """Per finished member: projected-season admissions (proj window) and its source run's fit-season total."""
    sc = pd.read_csv(exp.out_dir / "scores.csv").set_index("particle")
    rows = []
    for j in sorted((out_dir(exp) / mode / sid).glob("[ms][0-9]*.json")):
        meta = json.loads(j.read_text())
        w = pd.read_parquet(j.with_suffix(".parquet"))
        a, b = proj_window(exp)
        pw = w[(w.date >= pd.Timestamp(a)) & (w.date <= pd.Timestamp(b))]
        wk = pw.groupby("date").admissions.sum()
        rows.append({"member": meta["member"], "run_grouping": meta["run_grouping"], "source": meta["source"],
                     "fit_model_total": float(sc.loc[meta["source"], "fit_model_total"]),
                     "proj_model_total": float(wk.sum()), "proj_model_peak": float(wk.max()) if len(wk) else 0.0,
                     "proj_model_peak_week": wk.idxmax() if len(wk) else pd.NaT, **meta["s0"]})
    return pd.DataFrame(rows)


def season_record(exp: Experiment) -> pd.DataFrame:
    """The state's past NHSN seasons (GUARD_SEASONS), each over the projection window's calendar span: season,
    total, peak, winter_weeks, complete (>= GUARD_WINTER_WEEKS reported weeks Nov 15 .. Mar 15). 2024-25 misses
    its Aug-Oct weeks (voluntary reporting), which barely move a season's total."""
    import surveillance_data as sd
    h = sd.load_nhsn(start="2022-08-01")
    o = h[(h.location == exp.state) & (h.age_group == "0-130")].set_index("date").value.dropna()
    a, b = (pd.Timestamp(x) for x in proj_window(exp))
    rows = []
    for season in GUARD_SEASONS:
        y = int(season[:4]) - a.year
        lo, hi = a + pd.DateOffset(years=y), b + pd.DateOffset(years=y)
        s = o[(o.index >= lo) & (o.index <= hi)]
        winter = s[(s.index >= pd.Timestamp(f"{lo.year}-11-15")) & (s.index <= pd.Timestamp(f"{lo.year + 1}-03-15"))]
        rows.append({"season": season, "total": float(s.sum()), "peak": float(s.max()) if len(s) else np.nan,
                     "winter_weeks": len(winter), "complete": len(winter) >= GUARD_WINTER_WEEKS})
    return pd.DataFrame(rows)


def taper(x: np.ndarray, top: float = GUARD_MAX, full: float = GUARD_FULL) -> np.ndarray:
    """1 up to x = full, cos^2 down to 0 at x = top (0.5 halfway): x is a size over the largest past season."""
    u = np.clip((np.asarray(x, dtype=float) - full) / (top - full), 0.0, 1.0)
    return np.where(u >= 1.0, 0.0, np.cos(0.5 * np.pi * u) ** 2)


def peak_ok(tt: pd.DataFrame, peak_after: str | None) -> np.ndarray:
    """True where the member's all-age weekly admissions peak (week-ending date) is on or after peak_after (None:
    every member). No state peaked before November in NHSN 2022-26 (earliest: SC, week ending 2022-11-12)."""
    if not peak_after:
        return np.ones(len(tt), dtype=bool)
    return (pd.to_datetime(tt.proj_model_peak_week) >= pd.Timestamp(peak_after)).to_numpy()


def guard_weights(exp: Experiment, tt: pd.DataFrame, top: float = GUARD_MAX,
                  peak_after: str | None = None) -> tuple[np.ndarray, dict]:
    """Per member (member_totals rows): min(taper(total / record total), taper(peak / record peak)), the record =
    the largest past season (total and peak separately; a season missing weeks only undercounts, so it counts
    too). Room above every past season, none past `top` x it. No season: weight 1 (and a warning). peak_after
    (a date): weight 0 for a member whose season peak comes before it (peak_ok; a hard cut, no taper). Returns
    (unnormalized weights, the record)."""
    rec = season_record(exp)
    c = rec[rec.total > 0]
    timing = peak_ok(tt, peak_after).astype(float)
    if c.empty:
        print(f"!! {exp.name}: no past NHSN season in {GUARD_SEASONS}; size guard off", flush=True)
        return timing, ({"peak_after": peak_after} if peak_after else {})
    it, ip = c.total.idxmax(), c.peak.idxmax()
    r = {"record_total": float(c.total[it]), "record_total_season": c.season[it],
         "record_peak": float(c.peak[ip]), "record_peak_season": c.season[ip],
         "record_incomplete": bool(not c.complete[it] or not c.complete[ip])}
    if r["record_incomplete"]:
        print(f"!! {exp.name}: record season missing winter weeks ({r['record_total_season']}/"
              f"{r['record_peak_season']}): the guard may be tight", flush=True)
    w = np.minimum(taper(tt.proj_model_total / r["record_total"], top),
                   taper(tt.proj_model_peak / r["record_peak"], top))
    if peak_after:
        r["peak_after"] = peak_after
    return np.asarray(w, dtype=float) * timing, r


def _window_scores(exp: Experiment, w: pd.DataFrame, ms: pd.DataFrame, o: pd.Series, a, b, prefix: str) -> dict:
    """One member on observed weeks [a, b]: NB of all-age and of by-subtype weekly admissions (the calibration's
    likelihood with the projection background), totals and peaks. {} where nothing was observed."""
    oo = o[(o.index >= pd.Timestamp(a)) & (o.index <= pd.Timestamp(b))].dropna()
    if not len(oo):
        return {}
    win = dataclasses.replace(exp, proj_start=str(a), proj_end=str(b))
    m = w.groupby("date").admissions.sum().reindex(oo.index).fillna(0.0)
    rw = recent_weights(exp, oo, prefix)
    return {f"{prefix}_nb_ll": calib.nb_loglik(oo.to_numpy(), calib.with_background(win, "proj", m.to_numpy()),
                                               exp.likelihood["nb_size"], calib.week_weights(exp, oo.to_numpy()) * rw),
            f"{prefix}_subtype_nb_ll": calib.subtype_nb_ll(win, "proj", oo, ms, rw),
            f"{prefix}_weeks": int(len(oo)), f"{prefix}_obs_total": float(oo.sum()),
            f"{prefix}_model_total": float(m.sum()),
            f"{prefix}_obs_peak_week": oo.idxmax().date().isoformat(),
            f"{prefix}_model_peak_week": m.idxmax().date().isoformat(),
            f"{prefix}_obs_peak": float(oo.max()), f"{prefix}_model_peak": float(m.max()),
            **_slope(exp, oo, calib.with_background(win, "proj", m.to_numpy()), prefix)}


def recent_weights(exp: Experiment, oo: pd.Series, prefix: str) -> np.ndarray | float:
    """Extra NB weight on the early window's last weeks: production["early_recent_weeks"] (default 0 = off) weeks
    weigh production["early_recent_weight"] x. The level of the weeks just before the cutoff says most about the
    weeks after it (FL 2026: x1.57/week in mid-September, x1.37 and x1.32 in the last two)."""
    k = int(exp.production.get("early_recent_weeks", 0))
    if prefix != "early" or k <= 0 or not len(oo):
        return 1.0
    x = np.ones(len(oo))
    x[-k:] = float(exp.production.get("early_recent_weight", 1.0))
    return x


def _slope(exp: Experiment, oo: pd.Series, mu: np.ndarray, prefix: str) -> dict:
    """The weekly log growth of the window's last early_slope_weeks + 1 weeks (production["early_slope_weeks"],
    default 4), observed and model (with the background, as the likelihood sees it), and a Gaussian log-likelihood
    of their difference (sd production["early_slope_sd"], default 0.1/week): the NB terms weigh each week's level,
    so a member can pass close to the last weeks' level while growing three times as fast (FL 2026: x1.4/week
    observed, x2.5-3 in the members the level picked)."""
    k = int(exp.production.get("early_slope_weeks", 4))
    if len(oo) < k + 1:
        return {}
    o, m = np.log(np.maximum(oo.to_numpy()[-k - 1:], 1.0)), np.log(np.maximum(mu[-k - 1:], 1.0))
    go, gm = (o[-1] - o[0]) / k, (m[-1] - m[0]) / k
    sd = float(exp.production.get("early_slope_sd", 0.1))
    return {f"{prefix}_obs_growth": float(go), f"{prefix}_model_growth": float(gm),
            f"{prefix}_slope_ll": float(-0.5 * ((gm - go) / sd) ** 2)}


def early_slope_weight(exp: Experiment) -> float:
    """Weight of the growth term (_slope) in early_score: production["early_slope_weight"], default 0 (off)."""
    return float(exp.production.get("early_slope_weight", 0.0))


def early_subtype_weight(exp: Experiment) -> float:
    """Weight of the by-subtype term in early_score: production["early_subtype_weight"], else the calibration's
    likelihood.subtype_weekly, else 0.3 (it06's)."""
    return float(exp.production.get("early_subtype_weight", exp.likelihood.get("subtype_weekly", 0.3)))


def all_members(exp: Experiment) -> pd.DataFrame:
    """members.csv plus the members a screening selection created (restart/screened.csv; restart/selected.csv,
    which a guard selection later narrows)."""
    parts = [load_members(exp)]
    for f in (out_dir(exp) / "restart" / "screened.csv", out_dir(exp) / "restart" / "selected.csv"):
        if f.exists():
            parts.append(pd.read_csv(f))
    return pd.concat(parts, ignore_index=True).drop_duplicates("member", keep="last")


def score_members(exp: Experiment, mode: str = "restart", sid: str = "B") -> pd.DataFrame:
    """Round 2: every finished member of scenario `sid` against NHSN -> proj_<name>/<mode>/<sid>_scores.csv.
      early_*  the projected season's weeks the hub lets us see (early_window: restart anchor .. data_end);
               early_score = early_nb_ll + early_subtype_weight x early_subtype_nb_ll (+ early_slope_weight x
               early_slope_ll, the last weeks' growth rate: _slope; off by default) ranks and weights the
               selection (select weights="early")
      proj_*   proj_window (a backtest's whole projected season; production: the observed part of it)
      post_*   the origin .. proj_end (backtests only): the out-of-sample part, for judging a selection rule
    plus the member's parameters, projected-season subtype shares (model and observed) and S0 on the origin."""
    sid = scenario_id(exp, sid)
    a, b = proj_window(exp)
    ea, eb = early_window(exp)
    obs = calib.observed(exp)
    o = obs[obs.age_group == "0-130"].set_index("date").value
    mix = calib.observed_subtype_mix(exp, a, b)
    mem = all_members(exp)
    sw, gw = early_subtype_weight(exp), early_slope_weight(exp)
    rows = []
    for j in sorted((out_dir(exp) / mode / sid).glob("*.json")):
        meta = json.loads(j.read_text())
        w = pd.read_parquet(j.with_suffix(".parquet"))
        ms = w.pivot_table(index="date", columns="subtype", values="admissions", aggfunc="sum")
        ww = w[(w.date >= pd.Timestamp(a)) & (w.date <= pd.Timestamp(b))]
        pm = ww.groupby("subtype").infections.sum().reindex(SUBTYPES)
        pm = pm / pm.sum() if pm.sum() > 0 else pm.fillna(1 / 3)
        r = {"member": meta["member"], **_window_scores(exp, w, ms, o, ea, eb, "early"),
             **_window_scores(exp, w, ms, o, a, b, "proj"),
             **_window_scores(exp, w, ms, o, exp.production["origin"], b, "post"),
             **{f"proj_model_share_{k}": float(pm[k]) for k in SUBTYPES},
             **{f"proj_obs_share_{k}": float(mix[k]) for k in SUBTYPES},
             **{k: v for k, v in meta["s0"].items() if k.startswith("S0")}}
        if "early_nb_ll" in r:
            r["early_score"] = r["early_nb_ll"] + sw * r["early_subtype_nb_ll"] + gw * r.get("early_slope_ll", 0.0)
        rows.append(r)
    sc = pd.DataFrame(rows)
    if sc.empty:
        return sc
    key = next((k for k in ("early_score", "proj_nb_ll") if k in sc), None)   # neither: no NHSN week (KS 2026-27)
    sc = mem.merge(sc, on="member")
    sc = sc.sort_values(key, ascending=False) if key else sc
    sc.to_csv(out_dir(exp) / mode / f"{sid}_scores.csv", index=False)
    return sc


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.sum() > 0 else 0.0


def tempered_weights(score: np.ndarray, target_ess: float) -> tuple[np.ndarray, float]:
    """Weights ~ exp(score / T), T chosen so the weights' effective sample size is target_ess: T = 1 is the
    likelihood itself (with ~8 weekly observations and NB size 20 it is sharp enough to collapse onto a few
    members), T -> inf is uniform. Tempering keeps the ensemble as wide as the evidence warrants rather than as
    narrow as one noisy realization suggests. Returns (normalized weights, T)."""
    x = np.asarray(score, dtype=float)
    x = np.where(np.isfinite(x), x, np.nanmin(x[np.isfinite(x)]) - 1e3 if np.isfinite(x).any() else 0.0)
    n = len(x)
    if target_ess >= n or np.ptp(x) == 0:
        return np.full(n, 1.0 / n), np.inf

    def w_at(t):
        v = np.exp((x - x.max()) / t)
        return v / v.sum()
    lo, hi = -6.0, 8.0                                    # log10 T
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ess(w_at(10 ** mid)) < target_ess else (lo, mid)
    t = 10 ** hi
    return w_at(t), float(t)


def pps_sample(w: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n indices without replacement, inclusion probability pi_i = min(1, c w_i) with sum pi = n (members whose
    share would pass 1 are taken for certain and c re-solved on the rest), by systematic sampling over a random
    order. Efraimidis-Spirakis (used until 2026-10-30) only approximates pi and, keeping a third of a pool,
    almost never took a weight-0.5 member while weight-1 members were left. Returns (indices, pi)."""
    w = np.asarray(w, dtype=float)
    pos = w > 0
    n = min(n, int(pos.sum()))
    cert = np.zeros(len(w), dtype=bool)
    while True:
        rest = pos & ~cert
        pi = np.where(cert, 1.0, np.where(rest, (n - cert.sum()) * w / w[rest].sum(), 0.0))
        over = rest & (pi > 1.0)
        if not over.any():
            break
        cert |= over
    order = rng.permutation(len(w))
    edges = np.concatenate([[0.0], np.cumsum(pi[order])])
    u = rng.random() + np.arange(n)
    picked = order[np.searchsorted(edges, u, side="right") - 1]
    return np.unique(picked), pi


def guard_mass(exp: Experiment, peak_after: str | None = None) -> dict:
    """The guard's view of the finished full scenario-B runs (restart mode): runs, the summed guard weight (the
    "mass" PPS keeps n members from: below n, every weight-1 member is kept for certain and the tapered ones get
    more than their share), the pass rate (mass / runs) and the top-up round reached (screened.csv)."""
    tt = member_totals(exp, "restart", scenario_id(exp, "B"))
    g = guard_weights(exp, tt, peak_after=peak_after)[0] if len(tt) else np.zeros(0)
    scr = out_dir(exp) / "restart" / "screened.csv"
    rnd = int(pd.read_csv(scr).get("topup_round", pd.Series([0])).max()) if scr.exists() else 0
    return {"runs": len(tt), "mass": float(g.sum()), "pass_rate": float(g.sum() / len(tt)) if len(tt) else np.nan,
            "passing": int((g > 0).sum()), "topup_round": rnd,
            "early_peaks": int((~peak_ok(tt, peak_after)).sum()) if len(tt) else 0}


def topup_size(exp: Experiment, n: int, margin: float = 1.25, max_runs: int | None = None,
               peak_after: str | None = None) -> int:
    """How many more screened members a full B pass needs for the guard mass to reach margin x n, at the pass rate
    seen so far (0 when it is there; capped so the B runs stay <= max_runs)."""
    m = guard_mass(exp, peak_after)
    if m["mass"] >= margin * n or not m["runs"]:
        return 0
    add = int(np.ceil((margin * n - m["mass"]) / max(m["pass_rate"], 0.02)))
    return max(0, min(add, max_runs - m["runs"])) if max_runs else add


def topup(exp: Experiment, n: int, seed: int | None = None) -> pd.DataFrame:
    """n more screened members, drawn like the first n (systematic resampling, with replacement, on the screen
    selection's early-season weights: screen/selection.csv) and NOT from the guard's survivors, whose parameters
    the guard would then weigh twice. They continue the s#### ids and run_grouping, carry topup_round = k, and are
    appended to restart/screened.csv; restart/selected.csv becomes the whole screened set again (for their B runs)
    and a guard selection already made is set aside (restart/selection_round<k-1>.csv), so the next guard
    selection runs over every B run."""
    r = out_dir(exp) / "restart"
    old = pd.read_csv(r / "screened.csv")
    if "topup_round" not in old:
        old["topup_round"] = 0
    k = int(old.topup_round.max()) + 1
    cand = pd.read_csv(out_dir(exp) / "screen" / "selection.csv")
    w = cand.weight.to_numpy() / cand.weight.sum()
    rng = np.random.default_rng([seed if seed is not None else exp.seed, k])
    u = (rng.random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w), u), len(cand) - 1)
    mem = all_members(exp).drop(columns=["parent", "topup_round"], errors="ignore").set_index("member")
    new = mem.loc[cand.member.to_numpy()[idx]].reset_index().rename(columns={"member": "parent"})
    new = new[[c for c in old.columns if c in new.columns and c not in ("member", "run_grouping", "topup_round")]]
    start = int(old.member.str[1:].astype(int).max()) + 1
    new.insert(0, "member", [f"s{start + j:04d}" for j in range(n)])
    new["run_grouping"] = np.arange(start + 1, start + n + 1)
    new["topup_round"] = k
    both = pd.concat([old, new], ignore_index=True)[old.columns]
    both.to_csv(r / "screened.csv", index=False)
    both.to_csv(r / "selected.csv", index=False)
    if (r / "selection.csv").exists():
        (r / "selection.csv").replace(r / f"selection_round{k - 1}.csv")
    print(f"{exp.name}: top-up round {k}: {n} members ({new.parent.nunique()} parameter sets) "
          f"s{start:04d}..s{start + n - 1:04d} -> {r / 'screened.csv'}", flush=True)
    return new


def _free(path: Path) -> Path:
    """path, or path with _2, _3 ... before the suffix when it exists (an archived selection is never overwritten)."""
    k, q = 1, path
    while q.exists():
        k += 1
        q = path.with_name(f"{path.stem}_{k}{path.suffix}")
    return q


def grow_open(exp: Experiment, n: int, peak_after: str | None = None) -> list[str] | None:
    """Reopen a finished guard selection (state_pipeline.py --grow / --peak-after): when restart/selection.csv
    exists and either keeps fewer than n or (peak_after) keeps a member whose season peak comes before
    peak_after, the kept members that still pass are saved to restart/kept_grow.csv (select(..., keep=...) keeps
    them all again, so their scenario A/C/D runs are reused; the failing ones drop out), the old selection set
    aside (restart/selection_grow.csv and selected_grow.csv, numbered _2, _3 ... when those exist) and
    restart/selected.csv reset to every screened member, so the B step can top up and the guard select again.
    Returns the members kept, or None when there is nothing to reopen (so a rerun after the new guard is a no-op)."""
    r = out_dir(exp) / "restart"
    if not (r / "selection.csv").exists() or not (r / "selected.csv").exists():
        return None
    kept = pd.read_csv(r / "selected.csv")
    ok = np.ones(len(kept), dtype=bool)
    if peak_after:
        tt = member_totals(exp, "restart", scenario_id(exp, "B")).set_index("member")
        ok = peak_ok(tt.reindex(kept.member).reset_index(), peak_after)
    if len(kept) >= n and ok.all():
        return None
    kept[ok][["member"]].to_csv(r / "kept_grow.csv", index=False)
    kept.to_csv(_free(r / "selected_grow.csv"), index=False)
    (r / "selection.csv").replace(_free(r / "selection_grow.csv"))
    pd.read_csv(r / "screened.csv").to_csv(r / "selected.csv", index=False)
    print(f"{exp.name}: guard selection reopened: {len(kept)} kept, {int((~ok).sum())} peak before {peak_after} "
          f"dropped -> {int(ok.sum())} stay, grow to {n}" if peak_after else
          f"{exp.name}: guard selection reopened: {len(kept)} kept -> grow to {n}", flush=True)
    return list(kept.member[ok])


def grow_kept(exp: Experiment) -> list[str] | None:
    f = out_dir(exp) / "restart" / "kept_grow.csv"
    return list(pd.read_csv(f).member) if f.exists() else None


def select(exp: Experiment, mode: str, n: int = 300, weights: str = "history", seed: int | None = None,
           target_ess: float | None = None, keep: list[str] | None = None,
           peak_after: str | None = None) -> pd.DataFrame:
    """Keep n members of an oversampled scenario-B pass.
      weights="history"  probability ~ calib.intensity_weights (the member's next/fit admissions ratio against the
                         CDC season-to-season record), without replacement
      weights="early"    probability ~ tempered early-season likelihood (tempered_weights, ESS = target_ess,
                         default n / 2); equal weights, with a warning, for a state with no NHSN week in the
                         window (KS and AR stopped reporting in June 2026). From a screen pass (mode="screen": runs cut at the origin) the n
                         members are drawn WITH replacement (systematic resampling) and become new members
                         s0000.. (parent = the screened member; the simulator has no seed, so a repeated
                         parameter set is a new trajectory) in restart/selected.csv, to run in full for every
                         scenario with `flu_proj.py run --selected`. From a full pass: without replacement.
      weights="guard"    after a full scenario-B pass of the screened members (mode="restart"): guard_weights, the
                         member's season total and peak against the state's largest past season (1 up to
                         GUARD_FULL x it, 0 at GUARD_MAX x it), n kept without replacement among weight > 0
                         (fewer if not enough: oversample the screen selection). restart/selected.csv becomes
                         the kept members; the screen selection stays in restart/screened.csv.
      weights="uniform"
    Without replacement = systematic PPS (pps_sample): each member's inclusion probability is proportional to its
    weight (capped at 1), so a weight-0.5 member is kept half as often as a weight-1 one whatever the oversample.
    keep (full passes only): members kept for certain (an earlier, smaller selection being grown: grow_open);
    the other n - len(keep) are drawn by PPS among the rest.
    Writes <mode>/selection.csv (every candidate, its score, weight, inclusion probability and copies) and the
    selected members."""
    rng = np.random.default_rng(seed if seed is not None else exp.seed)
    if weights == "early":
        sc = score_members(exp, mode, "B")
        if "early_score" in sc:
            target = target_ess or n / 2
            w, t = tempered_weights(sc.early_score.to_numpy(), target)
            cand = sc[["member", "source", "early_score", "early_nb_ll", "early_subtype_nb_ll",
                       "early_model_total", "early_obs_total",
                       *[c for c in ("early_slope_ll", "early_model_growth", "early_obs_growth") if c in sc]]
                      ].assign(weight=w)
        else:                           # the state reported no NHSN week in the window (KS, AR in 2026-27)
            print(f"!! {exp.name}: no NHSN weeks in the early window {early_window(exp)}: equal weights (the "
                  f"priors already took the HHS-region evidence; the guard still applies)", flush=True)
            t, target = np.nan, np.nan
            cand = sc[["member", "source"]].assign(weight=1.0 / len(sc), early_fallback="uniform")
    elif weights == "guard":
        t, target = np.nan, np.nan
        tt = member_totals(exp, mode, scenario_id(exp, "B"))
        g, rec = guard_weights(exp, tt, peak_after=peak_after)
        if not (g > 0).any():
            raise RuntimeError(f"{exp.name}: every member fails the guard ({GUARD_MAX} x the largest past season"
                               f"{', or peaks before ' + peak_after if peak_after else ''}) {rec}")
        if (g > 0).sum() < n:
            print(f"!! {exp.name}: only {(g > 0).sum()} of {len(tt)} members pass the guard; "
                  f"keeping those (oversample the screen selection for {n})", flush=True)
            n = int((g > 0).sum())
        cand = tt[["member", "source", "proj_model_total", "proj_model_peak", "proj_model_peak_week"]].assign(
            guard=g, weight=g / g.sum(), **rec)
    else:
        t, target = np.nan, np.nan
        tt = member_totals(exp, mode, scenario_id(exp, "B"))
        w = calib.intensity_weights(tt, list(tt.index)) if weights == "history" else np.ones(len(tt))
        cand = tt[["member", "source"]].assign(weight=np.asarray(w) / np.sum(w))
    if mode == "screen":
        u = (rng.random() + np.arange(n)) / n                          # systematic resampling
        idx = np.minimum(np.searchsorted(np.cumsum(cand.weight.to_numpy()), u), len(cand) - 1)
        cand["copies"] = np.bincount(idx, minlength=len(cand))
        mem = all_members(exp).set_index("member")
        s = mem.loc[cand.member.to_numpy()[idx]].reset_index().rename(columns={"member": "parent"})
        s.insert(0, "member", [f"s{k:04d}" for k in range(n)])
        s["run_grouping"] = np.arange(1, n + 1)
        p = out_dir(exp) / "restart" / "selected.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(p.with_name("screened.csv"), index=False)              # kept when a guard narrows selected.csv
    else:
        if len(cand) < n:
            raise RuntimeError(f"{exp.name}: only {len(cand)} scenario-B members finished, need >= {n}")
        if keep:
            forced = cand.member.isin(keep).to_numpy()
            rest = np.flatnonzero(~forced & (cand.weight.to_numpy() > 0))
            m = max(0, min(n - int(forced.sum()), len(rest)))
            incl = np.where(forced, 1.0, 0.0)
            chosen = list(cand.member.to_numpy()[forced])
            if m:
                wr = cand.weight.to_numpy()[rest]
                pick_r, incl_r = pps_sample(wr / wr.sum(), m, rng)
                chosen += list(cand.member.to_numpy()[rest][pick_r])
                incl[rest] = incl_r
            keep = np.array(chosen)
        else:
            pick, incl = pps_sample(cand.weight.to_numpy(), n, rng)
            keep = cand.member.to_numpy()[pick]
        cand["inclusion"] = incl
        cand["copies"] = cand.member.isin(keep).astype(int)
        mem = all_members(exp)
        s = mem[mem.member.isin(keep)].reset_index(drop=True)
        p = out_dir(exp) / mode / "selected.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(p, index=False)
    cand.assign(temperature=t, target_ess=target, ess=ess(cand.weight)).to_csv(
        out_dir(exp) / mode / "selection.csv", index=False)
    print(f"{exp.name} [{mode}, {weights}]: {len(cand)} candidates, ESS {ess(cand.weight):.0f}"
          f"{f' (T {t:.3g})' if np.isfinite(t) else ''}; {cand.copies.gt(0).sum()} distinct kept -> {p}", flush=True)
    return s


def collect(exp: Experiment, mode: str, members_: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weekly, s0): every finished trajectory of every scenario. weekly: scenario_id, member, run_grouping,
    date, subtype, age_group, infections, admissions, deaths. s0: scenario_id, member, run_grouping, S0*."""
    ws, ss = [], []
    for sid in scenario_ids(exp):
        for j in sorted((out_dir(exp) / mode / sid).glob("[ms][0-9]*.json")):
            meta = json.loads(j.read_text())
            if members_ is not None and meta["member"] not in members_:
                continue
            ws.append(pd.read_parquet(j.with_suffix(".parquet")).assign(scenario_id=sid,
                                                                        run_grouping=meta["run_grouping"]))
            ss.append({"scenario_id": sid, "member": meta["member"], "run_grouping": meta["run_grouping"],
                       **meta["s0"]})
    return (pd.concat(ws, ignore_index=True) if ws else pd.DataFrame()), pd.DataFrame(ss)


# --------------------------------------------------------------------------------------
# pilot-job tasks (scripts/proj/flu_proj.py submits these; workers import them by name)
# --------------------------------------------------------------------------------------

def prepare_task(name: str, sid: str, mode: str) -> str:
    return str(prepare(load(name), sid, mode))


def run_task(name: str, sid: str, member: dict, mode: str) -> str:
    if member_path(load(name), mode, scenario_id(load(name), sid), member["member"]).with_suffix(".json").exists():
        return "exists"
    return str(run_member(name, sid, member, mode))
