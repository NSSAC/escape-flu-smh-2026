#!/usr/bin/env python
"""Experiments for an in-season forecasting study: <family>_<ST> from a production experiment (default prodv3_<ST>:
model v4, the submitted vaccine, tau x1.16, hosp x1.08), one scenario, a forecast calendar, widened next-season
priors for the particle pool.

    python scripts/forecast/make_fc_experiments.py VA MA WA            # -> experiments/fc2627_<ST>.json
    python scripts/forecast/make_fc_experiments.py VA --base prodv3 --family fc2627 --dry-run

What changes from the base (everything else, incl. the calibration's fixed values and searched priors, is kept, so
members.csv draws its searched parameters from the base's 2025-26 calibration: the study links
calib_<family>_<ST> -> the base's calib dir):
  end, production.end, proj_window    the study's proj_end (2027-05-01)
  production.scenarios                {"B": the study's coverage}: one scenario, business as usual
  campaigns                           the 2026-27 campaign on that coverage
  production.data_end                 the pool's pre-screen cutoff (--screen-data-end): only the loose pre-screen
                                      (fc.py prescreen) reads it; each forecast has its own cutoff
  production.screen_end               the last day of a screen run (--screen-end; default screen-data-end + 7)
  production.origin                   KEPT (2026-10-04): it is where restart_ltimm reads the calibration run's S0
  production.next                     tau_next_mult lower bound --tau-lo (0.4; Round 1 0.6), next-season seasonal
                                      peak upper bound + --peak-later days (28): prodv3's B runs were 2-4x NHSN
                                      by 2026-11-07 (median), so the pool needs room for slower seasons
  root, scratch                       the study directory and its scratch (experiment.py)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"


def make(st: str, a) -> dict:
    study = json.loads(Path(a.study).read_text())
    root = str(Path(a.study).resolve().parent)
    base = json.loads((EXPERIMENTS / f"{a.base}_{st}.json").read_text())
    e = copy.deepcopy(base)
    end = study["calendar"]["proj_end"]
    sc = study["scenario"]
    p = e["production"]
    screen_end = a.screen_end or str((pd.Timestamp(a.screen_data_end) + pd.Timedelta(days=7)).date())
    e["name"] = f"{a.family}_{st}"
    e["campaigns"] = [dict(c, coverage=sc["coverage"]) if c["season"] == p["season"] else c for c in e["campaigns"]]
    p["scenarios"] = {sc["id"]: sc["coverage"]}
    p["end"] = end
    p["proj_window"] = [p["proj_window"][0], end]
    p["data_end"] = a.screen_data_end
    p["screen_end"] = screen_end
    nxt = p["next"]
    lo, hi, kind = nxt["tau_next_mult"]
    nxt["tau_next_mult"] = [min(lo, a.tau_lo), hi, kind]
    lo, hi, kind = nxt["seas_peak_tick_next"]
    nxt["seas_peak_tick_next"] = [lo, round(hi + a.peak_later, 1), kind]
    e["root"] = root
    e["scratch"] = study["scratch"] + "/calib"
    for k in ("prodv3_base", "prodv5_from"):
        p.pop(k, None)
    e.setdefault("provenance", {})["forecast"] = {
        "generator": "escape-flu-smh/scripts/forecast/make_fc_experiments.py", "base": base["name"],
        "study": study["name"], "tau_next_mult": f"lower bound {a.tau_lo} (base {base['production']['next']['tau_next_mult'][0]})",
        "seas_peak_tick_next": f"upper bound +{a.peak_later} d", "screen": f"pre-screen on NHSN to {a.screen_data_end}, "
        f"screen runs to {screen_end}"}
    e["notes"] = (f"{study['name']}: particle pool for in-season forecasting from {base['name']} (same calibration, "
                  f"model {e['model']}), scenario {sc['id']} = {sc['coverage']}, runs to {end}. "
                  "make_fc_experiments.py; flu-forecasting docs/forecasting.md.")
    return e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+")
    ap.add_argument("--study", default="/project/bii_nssac/biocomplexity/SMH/flu-forecasting-FluTURE-2026-27/study.json")
    ap.add_argument("--base", default="prodv3")
    ap.add_argument("--family", default="fc2627")
    ap.add_argument("--screen-data-end", default="2026-10-24")
    ap.add_argument("--screen-end", default=None)
    ap.add_argument("--tau-lo", type=float, default=0.4)
    ap.add_argument("--peak-later", type=float, default=28.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for st in a.states:
        e = make(st, a)
        out = EXPERIMENTS / f"{e['name']}.json"
        if a.dry_run:
            print(json.dumps({k: e[k] for k in ("name", "end", "root", "scratch", "campaigns")}, indent=1))
            print(json.dumps({k: v for k, v in e["production"].items() if k != "next"}, indent=1))
            print("next tau/peak:", e["production"]["next"]["tau_next_mult"], e["production"]["next"]["seas_peak_tick_next"])
            continue
        if out.exists() and not a.force:
            print(f"{out} exists (--force to replace)")
            continue
        out.write_text(json.dumps(e, indent=1) + "\n")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
