"""In-sim vaccination schedule: hub coverage curves x the person file's ages -> vax_schedule rows.

Row = one band, one week: tick (the Sunday the week starts, relative to tick 0), new_vax_count, min_age,
max_age (exclusive), vax_campaign, vax_formulation_tick, propensity_max (the ESL's columns).

  doses     the week's increase in cumulative coverage x the number of agents in the band. Synthetic ages
            are integers, so "6 months - 4 years" is ages 1-4 (age 0 is ineligible) and the band's
            coverage applies to those agents.
  timing    a dose in the week ending Saturday S is given on the Sunday that week starts (S - 6). Weeks
            before tick 0 are dropped -- those doses belong to the person file's history.
  targeting propensity_max = Phi^-1(min(slack x final coverage, 0.999)): only agents whose latent
            vax_propensity is below it are eligible, so a campaign lands on the habitual vaccinators the
            history already vaccinated (vax_history.py note 3). slack = 1 is the strictest.
  formulation  flu_ic.FORMULATION[season] (WHO Northern Hemisphere composition announcement).
  scenarios a campaign {"season": "2026-27", "coverage": "2026-27-B", ...} takes its curve from another
            flu_data.COVERAGE key than its season; "coverage": null is no campaign (scenario D).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .experiment import FLU_ROUND, Experiment

sys.path.insert(0, str(FLU_ROUND / "notebooks"))
import flu_data as fd          # noqa: E402
import flu_ic                  # noqa: E402

#: coverage band -> integer synthetic ages [min_age, max_age)
BAND_AGES = {"0.5-4": (1, 5), "5-12": (5, 13), "13-17": (13, 18), "18-49": (18, 50),
             "50-64": (50, 65), "65+": (65, 120)}


def band_counts(person: pd.DataFrame) -> pd.Series:
    a = person.age.to_numpy()
    return pd.Series({b: int(((a >= lo) & (a < hi)).sum()) for b, (lo, hi) in BAND_AGES.items()})


def build(exp: Experiment, person: pd.DataFrame | None = None) -> pd.DataFrame:
    """The experiment's full schedule, every campaign, as a DataFrame in the ESL's column order."""
    person = person if person is not None else pd.read_parquet(exp.person_path, columns=["age"])
    n = band_counts(person)
    rows = []
    for c in exp.campaigns:
        season, cid = c["season"], int(c["campaign"])
        key = c.get("coverage", season)          # a scenario's curve ("2026-27-B"); None = no campaign
        if key is None:
            continue
        cov = fd.load_coverage(key)
        cov = cov[cov.geography == exp.state]
        if cov.empty:
            raise ValueError(f"no {key} coverage for {exp.state}")
        ftick = exp.tick(flu_ic.FORMULATION[season])
        for band, g in cov.groupby("age_group"):
            g = g.sort_values("date")
            p_final = float(g.coverage.iloc[-1])
            pmax = float(ndtri(min(max(exp.propensity_slack * p_final, 1e-6), 0.999)))
            cum = np.round(g.coverage.to_numpy() * n[band]).astype(int)       # cumulative doses
            for date, d in zip(g.date.iloc[1:], np.diff(cum)):
                tick = exp.tick(date - pd.Timedelta(days=6))
                if d <= 0 or tick < 0 or tick >= exp.num_ticks:
                    continue
                lo, hi = BAND_AGES[band]
                rows.append({"tick": tick, "new_vax_count": int(d), "min_age": lo, "max_age": hi,
                             "vax_campaign": cid, "vax_formulation_tick": ftick,
                             "propensity_max": round(pmax, 4), "season": season, "band": band})
    cols = ["tick", "new_vax_count", "min_age", "max_age", "vax_campaign", "vax_formulation_tick",
            "propensity_max", "season", "band"]
    return pd.DataFrame(rows, columns=cols).sort_values(["tick", "vax_campaign", "min_age"]).reset_index(drop=True)


def esl_rows(sched: pd.DataFrame) -> list[dict]:
    cols = ["tick", "new_vax_count", "min_age", "max_age", "vax_campaign", "vax_formulation_tick",
            "propensity_max"]
    return [dict(zip(cols, r)) for r in sched[cols].itertuples(index=False)]
