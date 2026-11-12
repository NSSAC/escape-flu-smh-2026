"""Severity replicates: more trajectories from finished runs, by post-processing their admissions (2026-11-04).

    p = upscale.Params(k=5)                                   # the original + 4 replicates per member
    anchor = upscale.DeathAnchor.from_model(exp, sd_table)    # the model's P(death | hosp) and delay
    w5 = upscale.upscale(w, exp.state, p, anchor, cutoff="2026-10-03")

WHAT A REPLICATE IS
  The member's own epidemic (infections untouched) with its hospitalization severity and its deaths re-drawn.
  Admissions are scaled, not recomputed from infections: a run's admissions / infections already carry the
  model's protection against hospitalization from prior infection and from vaccination, so the A-D contrasts
  keep the vaccine's effect against hospitalization. For replicate r of member m (r = 0 is the run as simulated):

    admissions'[s, a, t] = admissions[s, a, t] x c x exp(z_sub[s] + z_age[a] + u_state[s]) x exp(x[t])
    deaths'[s, a, t]     = sum_k admissions'[s, a, t - k] x kernel[k] x p_death[s, a] x exp(z_death[s, a])

  z_sub, z_age    national log multipliers per subtype and per age band (sd_sub, sd_age)
  u_state         a state's own log multiplier per subtype (sd_state)
  c               rescales the draw so the member's scenario-B admissions up to the fitting cutoff keep their
                  total (anchor_fit): the data through Oct 3 pin the current wave's severity x infections, so a
                  replicate may move the age shape and the other subtypes' severity, not its fit
  x[t]            weekly AR(1) on log admissions (rho, stationary sd noise_sd), 0 up to the cutoff, then growing
                  from 0: national and state paths mixed by noise_state_frac (share of the variance)
  p_death         the model's in-hospital P(death | hospitalized) (death_hosp_<sub>_<band> x death_prob_scale,
                  read from the ESL and the experiment), z_death its national log perturbation (sd from the
                  severity table's 95% interval, at least sd_death_floor)
  kernel          the model's hosp_to_death delay, as week offsets (admissions spread evenly over their week)

  Replicate 0 keeps its simulated admissions and gets deaths' with no perturbation (the expected value of the
  simulated in-hospital deaths), so every replicate's deaths are made the same way.

ALL DEATHS (Params.all_deaths, 2026-11-05)
  The hub's death target is CDC's in-season multiplier model (every flu death, in and out of hospital). With an
  all-deaths table (DeathAnchor.from_model(..., all_table=); flu-round1-2026-27 notebooks/all_deaths_ratio.py)
  every trajectory's in-hospital deaths are further scaled, after the convolution:

    all_deaths'[s, a, t] = deaths'[s, a, t] x q[a] / p_pool[a] x exp(z_level + z_age[a])

  q[a]            all deaths per NHSN admission by age (CDC in-season estimates / NHSN, age profile from CDC
                  burden deaths / hospitalizations), p_pool[a] the model's pooled P(death | hosp) x death_prob_scale
  z_level, z_age  national log perturbations (log_sd_level, log_sd_age), drawn for every trajectory r = 0
                  included, from their own stream keyed by the trajectory position: admissions and every other
                  draw are the same with and without it, and every scenario and state shares them

HOW MANY
  Params.k replicates per run (the run included), or Params.target trajectories per state: k = ceil(target /
  runs), and when that overshoots the replicates are dealt round by round (every run's r = 0, then every run's
  r = 1, ...) until target, so a run has k - 1 or k trajectories (trajectories(); 60 runs x 5 = 300 exactly).

IDS AND PAIRING
  Trajectory id <member>r<r>: the run (member, its run_grouping) and the replicate r; the hub file writes them as
  run_grouping = the member's run_grouping and stochastic_run = r + 1, and every target of a trajectory (its
  admissions by age, its deaths, its S0) carries that pair. A replicate uses the same draws in every scenario
  (keyed by member and replicate, never by scenario; c is computed on scenario B), so scenario contrasts stay
  paired. National draws are keyed by the trajectory's position in the state's sorted id list, so the US sum's
  i-th trajectory (index pairing: flu_analysis.national, smh_submission) has the same national draw in every
  state; with the same runs x replicates in every state the i-th trajectory is the same (run index, r) too.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import SUBTYPES, Experiment

AGES = ["0-4", "5-17", "18-49", "50-64", "65-130"]
#: output age band -> the model's death band (death_hosp_<sub>_<band>)
DEATH_BAND = {"0-4": "0_17", "5-17": "0_17", "18-49": "18_49", "50-64": "50_64", "65-130": "65p"}
#: severity table's age labels (subtype_severity: death_given_hosp_by_subtype_age.csv)
TABLE_BAND = {"0_17": "0-17", "18_49": "18-49", "50_64": "50-64", "65p": "65-130"}


@dataclass
class Params:
    k: int = 5                      # trajectories per member, the original included (unless target is set)
    target: int | None = None       # trajectories per state: k = ceil(target / runs), dealt round by round
    seed: int = 20261104
    sd_sub: float = 0.10            # national log-sd per subtype
    sd_age: float = 0.10            # national log-sd per age band
    sd_state: float = 0.05          # state log-sd per subtype
    noise_sd: float = 0.10          # weekly AR(1) on log admissions after the cutoff: stationary sd
    noise_rho: float = 0.8          # its week-to-week correlation
    noise_state_frac: float = 0.5   # share of the noise variance that is the state's own
    sd_death_floor: float = 0.10    # least log-sd of P(death | hosp)
    anchor_fit: bool = True         # keep each replicate's admissions to the cutoff at the member's total
    all_deaths: bool = False        # scale deaths to CDC all deaths (the anchor needs all_table; see ALL DEATHS)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeathAnchor:
    p: dict                                    # (subtype, age band) -> P(death | hospitalized)
    sd: dict                                   # (subtype, age band) -> log-sd of its perturbation
    kernel: np.ndarray                         # weekly offsets 0, 1, ...: share of deaths in week t + k
    source: dict = field(default_factory=dict)
    all_mult: np.ndarray | None = None         # by age (AGES): all deaths / in-hospital deaths (ALL DEATHS)
    all_sd_level: float = 0.0                  # national log-sd of the all-deaths level
    all_sd_age: np.ndarray | None = None       # by age: log-sd of the age profile

    @classmethod
    def from_model(cls, exp: Experiment, sd_table: str | Path | None = None, floor: float = 0.10,
                   all_table: str | Path | None = None) -> "DeathAnchor":
        text = Path(exp.model_path).read_text()
        cfg = {k: float(v) for k, v in re.findall(r"^config\s+(\w+)\s*:\s*float\s*=\s*([-0-9.eE+]+)", text, re.M)}
        cfg.update({k: float(v) for k, v in exp.fixed.items() if k.startswith("death_")})
        scale = cfg.get("death_prob_scale", 1.0)
        p = {(s, a): min(cfg[f"death_hosp_{s}_{DEATH_BAND[a]}"] * scale, 1.0) for s in SUBTYPES for a in AGES}
        sd = {key: floor for key in p}
        if sd_table is not None and Path(sd_table).exists():
            t = pd.read_csv(sd_table).set_index(["age_group", "subtype"])
            for s in SUBTYPES:
                for a in AGES:
                    key = (TABLE_BAND[DEATH_BAND[a]], s)
                    if key in t.index:
                        lo, hi = t.loc[key, ["p_lo", "p_hi"]]
                        sd[(s, a)] = max(floor, float(np.log(hi / lo) / 3.92) if lo > 0 and hi > lo else 0.0)
        src = {"model": Path(exp.model_path).name, "death_prob_scale": scale,
               "sd_table": str(sd_table) if sd_table else None}
        out = cls(p, sd, delay_kernel(text), src)
        if all_table is not None:
            if sd_table is None:
                raise ValueError("all_table needs sd_table (its p_death_all_subtypes is the pooled in-hospital ratio)")
            pool = pd.read_csv(sd_table).drop_duplicates("age_group").set_index("age_group").p_death_all_subtypes
            t = pd.read_csv(all_table).set_index("age_group").reindex(AGES)
            p_pool = np.array([float(pool[TABLE_BAND[DEATH_BAND[a]]]) * scale for a in AGES])
            out.all_mult = t.q.to_numpy(float) / p_pool
            out.all_sd_level = float(t.log_sd_level.iloc[0])
            out.all_sd_age = t.log_sd_age.to_numpy(float)
            src["all_table"] = str(all_table)
        return out


def delay_kernel(esl_text: str, name: str = "hosp_to_death") -> np.ndarray:
    """The ESL discrete distribution `name` (days) as weekly offsets, admissions spread evenly over the week."""
    m = re.search(rf"discrete\s+{name}\s*\n(.*?)\n\s*end", esl_text, re.S)
    pv = [(float(p), int(float(v))) for p, v in re.findall(r"p\s*=\s*([0-9.eE+-]+)\s*,\s*v\s*=\s*([0-9.]+)",
                                                            m.group(1))]
    tot = sum(p for p, _ in pv)
    out = np.zeros(1 + (6 + max(v for _, v in pv)) // 7)
    for u in range(7):
        for p, v in pv:
            out[(u + v) // 7] += p / tot / 7
    return out


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng([int(k) if isinstance(k, (int, np.integer)) else zlib.crc32(str(k).encode())
                                  for k in key])


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """A unit-stationary-sd AR(1) path of length n that starts from 0."""
    x, e = np.zeros(n), rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    prev = 0.0
    for t in range(n):
        prev = rho * prev + e[t]
        x[t] = prev
    return x


def trajectories(members: list[str], p: Params) -> list[tuple[str, int]]:
    """(member, r) of every trajectory, sorted by id <member>r<r> (see HOW MANY)."""
    members = sorted(members)
    if p.target is None:
        pairs = [(m, r) for m in members for r in range(p.k)]
    else:
        k = int(np.ceil(p.target / len(members))) if members else 0
        pairs = [(m, r) for r in range(k) for m in members][:p.target]
    return sorted(pairs, key=lambda t: traj_id(*t))


def traj_id(member: str, r: int) -> str:
    return f"{member}r{r}"


def split_id(tid: str) -> tuple[str, int]:
    m, _, r = tid.rpartition("r")
    return m, int(r)


def draws(state: str, pos: int, member: str, r: int, p: Params, anchor: DeathAnchor, n_after: int) -> dict:
    """Replicate r's draws (all zero for r = 0). National parts keyed by the trajectory's position pos in the
    state's sorted id list, the state's own by (state, member, r)."""
    ns, na = len(SUBTYPES), len(AGES)
    if r == 0:
        return {"log_m": np.zeros((ns, na)), "noise": np.zeros(n_after), "log_d": np.zeros((ns, na))}
    g_nat, g_st = _rng(p.seed, "national", pos), _rng(p.seed, "state", state, member, r)
    z_sub = g_nat.normal(0, p.sd_sub, ns)
    z_age = g_nat.normal(0, p.sd_age, na)
    sd_d = np.array([[anchor.sd[(s, a)] for a in AGES] for s in SUBTYPES])
    # one national death draw per (subtype, death band): 0-4 and 5-17 share the model's 0-17 value
    bands = sorted(set(DEATH_BAND.values()))
    zb = g_nat.standard_normal((ns, len(bands)))
    z_d = np.array([[zb[i, bands.index(DEATH_BAND[a])] for a in AGES] for i in range(ns)]) * sd_d
    nat = _ar1(g_nat, n_after, p.noise_rho)
    u_state = g_st.normal(0, p.sd_state, ns)
    own = _ar1(g_st, n_after, p.noise_rho)
    f = p.noise_state_frac
    return {"log_m": z_sub[:, None] + z_age[None, :] + u_state[:, None],
            "noise": p.noise_sd * (np.sqrt(1 - f) * nat + np.sqrt(f) * own), "log_d": z_d}


def all_deaths_mult(pos: int, p: Params, anchor: DeathAnchor) -> np.ndarray:
    """By age (AGES): trajectory pos's all deaths / in-hospital deaths (ALL DEATHS), from its own national stream."""
    if anchor.all_mult is None:
        raise ValueError("Params.all_deaths needs DeathAnchor.from_model(..., all_table=)")
    g = _rng(p.seed, "all_deaths", pos)
    z = g.normal(0, anchor.all_sd_level) + g.standard_normal(len(AGES)) * anchor.all_sd_age
    return anchor.all_mult * np.exp(z)


def upscale(w: pd.DataFrame, state: str, p: Params, anchor: DeathAnchor, cutoff: str,
            ref_scenario: str = "B") -> pd.DataFrame:
    """w: scenario, member, date, subtype, age_group, infections, admissions[, deaths] (weekly, one state) ->
    the same columns for k replicates per member, deaths post-processed (see the module doc)."""
    cutoff = pd.Timestamp(cutoff)
    scen = sorted(w.scenario.unique())
    members = sorted(w.member.unique())
    dates = pd.DatetimeIndex(sorted(w.date.unique()))
    idx = pd.MultiIndex.from_product([scen, members, SUBTYPES, AGES, dates],
                                     names=["scenario", "member", "subtype", "age_group", "date"])
    have = w.groupby(["scenario", "member"]).size()                         # (scenario, member) that ran
    g = w.set_index(["scenario", "member", "subtype", "age_group", "date"])
    shape = (len(scen), len(members), len(SUBTYPES), len(AGES), len(dates))
    H = g.admissions.groupby(level=list(range(5))).sum().reindex(idx, fill_value=0).to_numpy(float).reshape(shape)
    I = g.infections.groupby(level=list(range(5))).sum().reindex(idx, fill_value=0).to_numpy(float).reshape(shape)
    after = dates > cutoff
    n_after = int(after.sum())
    pd_ = np.array([[anchor.p[(s, a)] for a in AGES] for s in SUBTYPES])
    kern = anchor.kernel
    si = scen.index(next(s for s in scen if s.startswith(ref_scenario))) if any(
        s.startswith(ref_scenario) for s in scen) else 0
    out = []
    col = {m: j for j, m in enumerate(members)}
    for pos, (m, r) in enumerate(trajectories(members, p)):
        j = col[m]
        d = draws(state, pos, m, r, p, anchor, n_after)
        mult = np.exp(d["log_m"])                                       # (sub, age)
        if p.anchor_fit and r > 0:
            pre = H[si, j][:, :, ~after]
            tot = pre.sum()
            c = tot / (pre * mult[:, :, None]).sum() if tot > 0 else 1.0
            mult = mult * c
        noise = np.ones(len(dates))
        noise[after] = np.exp(d["noise"])
        h = H[:, j] * mult[None, :, :, None] * noise[None, None, None, :]   # (scen, sub, age, date)
        conv = np.zeros_like(h)
        for kk, wk in enumerate(kern):
            if kk == 0:
                conv += wk * h
            else:
                conv[..., kk:] += wk * h[..., :-kk]
        dth = conv * (pd_ * np.exp(d["log_d"]))[None, :, :, None]
        if p.all_deaths:
            dth = dth * all_deaths_mult(pos, p, anchor)[None, None, :, None]
        for i, s in enumerate(scen):
            if (s, m) not in have.index:
                continue
            out.append(pd.DataFrame({
                "scenario": s, "member": traj_id(m, r),
                "date": np.tile(dates, len(SUBTYPES) * len(AGES)),
                "subtype": np.repeat(SUBTYPES, len(AGES) * len(dates)),
                "age_group": np.tile(np.repeat(AGES, len(dates)), len(SUBTYPES)),
                "infections": I[i, j].ravel(), "admissions": h[i].ravel(), "deaths": dth[i].ravel()}))
    return pd.concat(out, ignore_index=True)


def expand_members(t: pd.DataFrame, ids: list[str], col: str = "member") -> pd.DataFrame:
    """Copy a per-member table (S0, fits) to every trajectory id in ids (<member>r<r>)."""
    base = pd.DataFrame({"_tid": ids, col: [split_id(i)[0] for i in ids]})
    return t.merge(base, on=col).drop(columns=col).rename(columns={"_tid": col})


def replicates_of(p: Params, runs: int) -> int:
    """Trajectories per run: k, or with a target the largest per-run count."""
    return p.k if p.target is None else int(np.ceil(p.target / max(runs, 1)))
