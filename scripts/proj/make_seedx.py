#!/usr/bin/env python
"""seedx2_<ST>: a state's preview experiment (prod2025o_<ST>) with the FL early-season fix (2026-11-02), reusing
the preview's round 1 so only the next-season stage runs.

    python scripts/proj/make_seedx.py WA LA CA          # experiments/seedx2_<ST>.json + calib_seedx2_<ST>/
    python scripts/proj/state_pipeline.py submit run seedx2_WA seedx2_LA seedx2_CA --tag early-rise ...

The fix (seedx2_FL; see docs/flu.md, "FL early-season test"): model v3, the next-season A_H1N1 seeding as a pulse
(seed_boost_A_H1N1_next 1-30x the fitted rate for seed_days_A_H1N1_next 7-35 days from seed_start_A_H1N1_next,
Aug 8 - Sep 12), escape_next_A_H1N1 up to 0.45, and the early score's growth term at sd 0.03/week. Everything
else (priors, fixed values, seed) is the state's prod2025o experiment.

Round 1 is copied from calib_prod2025o_<ST> with the explore results re-keyed to the new name, so the state must
have finished its preview explore; states that have not are skipped (rerun when they have).
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py-modules"))
from flu_smh import experiment  # noqa: E402

BASE, NAME = "prod2025o_{}", "seedx2_{}"
MODEL = "flu-r1-2026-27-v3.esl"
NEXT = {"seed_start_A_H1N1_next": [364.0, 399.0, "lin"],
        "seed_boost_A_H1N1_next": [1.0, 30.0, "log"],
        "seed_days_A_H1N1_next": [7.0, 35.0, "lin"],
        "escape_next_A_H1N1": [0.0, 0.45, "lin"]}
PRODUCTION = {"early_slope_weight": 1.0, "early_slope_sd": 0.03}
POINTS = 64


def make(st: str, force: bool = False) -> str:
    base, name = BASE.format(st), NAME.format(st)
    src = experiment.EXPERIMENTS / f"{base}.json"
    d = json.loads(src.read_text())
    e = experiment.load(base)
    pk = e.out_dir / "sw" / f"explore-s{e.seed}-n{POINTS}.pkl.gz"
    if not pk.exists():
        return f"{st}: skipped, {base} has not finished its explore ({pk.name} missing)"
    with gzip.open(pk, "rb") as f:
        res = pickle.load(f)
    if len(res[base]["values"]) < POINTS:
        return f"{st}: skipped, {base}'s explore is incomplete"
    dst = experiment.EXPERIMENTS / f"{name}.json"
    if dst.exists() and not force:
        return f"{st}: {name} exists (--force to rewrite)"
    d["name"], d["model"] = name, MODEL
    d["production"]["next"].update(NEXT)
    d["production"].update(PRODUCTION)
    d["notes"] = (f"Early-rise fix (2026-11-02): {base}'s round 1 (copied) and seed with seedx2_FL's next-season "
                  "A_H1N1 seeding pulse (model v3), H1N1 escape up to 0.45 and the early growth term at sd 0.03 "
                  "(scripts/proj/make_seedx.py). " + d.get("notes", ""))
    dst.write_text(json.dumps(d, indent=1))
    out = experiment.load(name).out_dir
    (out / "sw").mkdir(parents=True, exist_ok=True)
    for p in ("runs", "scores.csv"):
        s = e.out_dir / p
        if s.is_dir():
            shutil.copytree(s, out / p, dirs_exist_ok=True)
        elif s.exists():
            shutil.copy2(s, out / p)
    with gzip.open(out / "sw" / pk.name, "wb") as f:
        pickle.dump({name: res[base]}, f)
    return f"{st}: {name} -> {dst.name}, {out}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for st in a.states:
        print(make(st, a.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
