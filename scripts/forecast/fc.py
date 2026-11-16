#!/usr/bin/env python
"""In-season forecasting driver (flu_smh/forecast.py). Run from escape-flu-smh/ with the study's environment:

    source /project/bii_nssac/biocomplexity/SMH/flu-forecasting-FluTURE-2026-27/env.sh
    python scripts/forecast/fc.py status
    python scripts/forecast/fc.py snapshot                        # today's NHSN -> data/snapshots/nhsn/<today>/
    python scripts/forecast/fc.py forecast                        # latest snapshot, its last week -> forecasts/<ref>/
    python scripts/forecast/fc.py backtest --through 2026-10-17 2026-10-24 2026-10-31
    python scripts/forecast/fc.py score                           # every issued forecast vs the latest snapshot
    python scripts/forecast/fc.py weekly                          # snapshot + forecast + score + reports

Pool p01 (new particles; the owner submits the SLURM steps, see docs/forecasting.md):
    python scripts/proj/flu_proj.py members fc2627_VA fc2627_MA fc2627_WA --design sobol --top 16 --per-fit 64
    sbatch scripts/forecast/pool.sbatch fc2627_VA fc2627_MA fc2627_WA     # screen -> prescreen -> full runs
    python scripts/forecast/fc.py prescreen fc2627_VA ...                  # (pool.sbatch runs it)

Outputs (study directory = $FLU_ROUND):
    forecasts/<reference_date>/            live forecasts (data to reference_date - 7 from the newest snapshot)
    backtests/<snapshot>/<reference_date>/ retrospective forecasts from one (later) snapshot: NOT what a live
                                           forecast would have seen (NHSN revisions)
    forecasts/scores.csv, backtests/<snapshot>/scores.csv
    <dir>/report.html, <dir>/../index.html  (notebooks/forecast_report.py)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if "FLU_ROUND" not in os.environ or not (Path(os.environ["FLU_ROUND"]) / "study.json").exists():
    sys.exit("fc.py: FLU_ROUND must be a forecasting study's checkout (source <study>/env.sh)")
sys.path.insert(0, str(REPO / "py-modules"))

import pandas as pd  # noqa: E402

from flu_smh import experiment, forecast  # noqa: E402

ROOT = experiment.FLU_ROUND
NB = ROOT / "notebooks"


def report(paths: list[Path]) -> None:
    subprocess.run([sys.executable, str(NB / "forecast_report.py"), *map(str, paths)], check=True)


def cmd_status(a) -> int:
    S = forecast.study()
    print(f"study {S['name']}  ({ROOT})")
    snaps = sorted(p.name for p in (ROOT / "data/snapshots/nhsn").glob("20??-??-??"))
    for s in snaps:
        m = json.loads((ROOT / "data/snapshots/nhsn" / s / "meta.json").read_text())
        print(f"  snapshot {s}: last week {m['last_week']}")
    for pool, d in S["pools"].items():
        n = {st: len(forecast.pool_files(S, st, pool)) for st in S["locations"]}
        print(f"  pool {pool}: {n}{'  (in forecast.pools)' if pool in S['forecast']['pools'] else ''}")
    for st in S["locations"]:
        e = experiment.load(f"{S['experiment_family']}_{st}")
        pd_ = ROOT / f"proj_{e.name}"
        parts = {"members": pd_ / "members.csv", "screen runs": pd_ / "screen" / "B",
                 "prescreen": pd_ / "restart" / "selected.csv", "full runs": pd_ / "restart" / "B"}
        info = []
        for k, p in parts.items():
            if p.suffix == ".csv":
                info.append(f"{k} {len(pd.read_csv(p)) if p.exists() else '-'}")
            else:
                info.append(f"{k} {len(list(p.glob('*.parquet'))) if p.exists() else '-'}")
        print(f"  {e.name}: " + ", ".join(info))
    for d in sorted((ROOT / "forecasts").glob("20??-??-??")):
        m = json.loads((d / "manifest.json").read_text())
        print(f"  forecast {d.name}: data to {m['data_through']} (snapshot {m['snapshot']}), primary {m['primary']}")
    return 0


def cmd_snapshot(a) -> int:
    sys.path.insert(0, str(NB))
    import nhsn_snapshot
    nhsn_snapshot.make(a.as_of, a.force)
    return 0


def _last_week(as_of: str) -> str:
    return json.loads((forecast.snapshot_dir(as_of) / "meta.json").read_text())["last_week"]


def cmd_forecast(a) -> int:
    S = forecast.study()
    as_of = a.as_of or forecast.latest_snapshot()
    through = a.through or _last_week(as_of)
    out = ROOT / "forecasts" / str(forecast.reference_date(through).date())
    if (out / "manifest.json").exists() and not a.force:
        print(f"{out} exists (--force to redo it)")
        return 0
    print(f"forecast: data to {through} from snapshot {as_of} -> {out}")
    forecast.make(S, through, as_of, out, a.states, a.pools, kind="live")
    if not a.no_report:
        report([out])
    return 0


def cmd_backtest(a) -> int:
    S = forecast.study()
    as_of = a.as_of or forecast.latest_snapshot()
    base = ROOT / "backtests" / as_of
    outs = []
    for t in a.through:
        out = base / str(forecast.reference_date(t).date())
        if (out / "manifest.json").exists() and not a.force:
            print(f"{out} exists")
        else:
            print(f"backtest: data to {t} from snapshot {as_of} -> {out}")
            forecast.make(S, t, as_of, out, a.states, a.pools, kind="backtest")
        outs.append(out)
    sc = forecast.score(outs, as_of)
    sc.to_csv(base / "scores.csv", index=False)
    if not a.no_report:
        report([*outs, base])
    return 0


def cmd_score(a) -> int:
    truth = a.truth or forecast.latest_snapshot()
    dirs = sorted(d for d in (ROOT / "forecasts").glob("20??-??-??") if (d / "manifest.json").exists())
    if not dirs:
        print("no forecasts yet")
        return 0
    sc = forecast.score(dirs, truth)
    sc.to_csv(ROOT / "forecasts" / "scores.csv", index=False)
    print(f"{len(sc)} scored rows (truth {truth}) -> {ROOT / 'forecasts' / 'scores.csv'}")
    return 0


def cmd_prescreen(a) -> int:
    as_of = a.as_of or forecast.latest_snapshot()
    for name in a.experiments:
        forecast.prescreen(name, as_of, n_keep=a.n, band=a.band, weeks=a.weeks)
    return 0


def cmd_weekly(a) -> int:
    cmd_snapshot(argparse.Namespace(as_of=None, force=False))
    cmd_forecast(argparse.Namespace(as_of=None, through=None, states=None, pools=None, force=False, no_report=True))
    cmd_score(argparse.Namespace(truth=None))
    dirs = sorted(d for d in (ROOT / "forecasts").glob("20??-??-??") if (d / "manifest.json").exists())
    report([*dirs[-1:], ROOT / "forecasts"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    s = sp.add_parser("snapshot")
    s.add_argument("--as-of")
    s.add_argument("--force", action="store_true")
    for name in ("forecast", "backtest"):
        f = sp.add_parser(name)
        f.add_argument("--as-of", help="snapshot (default the newest)")
        f.add_argument("--states", nargs="*")
        f.add_argument("--pools", nargs="*", help="default study.json forecast.pools")
        f.add_argument("--force", action="store_true")
        f.add_argument("--no-report", action="store_true")
        if name == "forecast":
            f.add_argument("--through", help="last observed week (default the snapshot's last week)")
        else:
            f.add_argument("--through", nargs="+", required=True, help="last observed weeks (Saturdays)")
    s = sp.add_parser("score")
    s.add_argument("--truth", help="snapshot used as truth (default the newest)")
    s = sp.add_parser("prescreen")
    s.add_argument("experiments", nargs="+")
    s.add_argument("--as-of")
    s.add_argument("--n", type=int, default=384)
    s.add_argument("--band", type=float, default=4.0)
    s.add_argument("--weeks", type=int, default=3)
    sp.add_parser("weekly")
    a = ap.parse_args(argv)
    return {"status": cmd_status, "snapshot": cmd_snapshot, "forecast": cmd_forecast, "backtest": cmd_backtest,
            "score": cmd_score, "prescreen": cmd_prescreen, "weekly": cmd_weekly}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
