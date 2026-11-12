#!/usr/bin/env python
"""Calibrate / backtest models/flu-r1-2026-27.esl. One experiment = experiments/<name>.json.

    module load gcc/14.2.0 && conda activate escape
    python scripts/calib/flu_calib.py prepare bt2023_VA          # schedule, build, input.h5 (once)
    python scripts/calib/flu_calib.py sample  bt2023_VA          # Sobol particles -> particles.csv
    python scripts/calib/flu_calib.py run     bt2023_VA --ids 0 1 2
    python scripts/calib/flu_calib.py run     bt2023_VA --shard 3 --nshards 32   # under SLURM
    python scripts/calib/flu_calib.py submit  bt2023_VA --nshards 32 [--dry-run]
    python scripts/calib/flu_calib.py score   bt2023_VA          # -> scores.csv (ranked)
    python scripts/calib/flu_calib.py status  bt2023_VA

Small outputs go to flu-round1-2026-27/calib_<name>/ (particles.csv, runs/p####.parquet weekly tables,
scores.csv); the build, input.h5 and raw HDF5 go to /scratch/$USER/flu-round1-2026-27/calib/<name>/.
Runs are resumable: `run` skips particles whose weekly table exists unless --force.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py-modules"))
from flu_smh import calib, experiment, sim  # noqa: E402


def cmd_prepare(exp, a):
    t0 = time.time()
    exp.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(experiment.EXPERIMENTS / f"{exp.name}.json", exp.out_dir / "experiment.json")
    sim.prepare(exp, build=not a.no_build)
    s = sim.pd.read_csv(sim.schedule_path(exp))
    print(f"prepared {exp.name} in {time.time() - t0:.0f}s: {exp.num_ticks} ticks, "
          f"{len(s)} vaccination rows, {s.new_vax_count.sum():,} doses -> {exp.work_dir}")


def cmd_sample(exp, a):
    p = exp.out_dir / "particles.csv"
    if p.exists() and not a.force:
        print(f"{p} exists (use --force to resample)")
        return
    exp.out_dir.mkdir(parents=True, exist_ok=True)
    df = calib.sample(exp, a.n)
    df.to_csv(p, index=False)
    print(f"{len(df)} particles x {df.shape[1] - 1} parameters -> {p}")


def _ids(exp, a):
    import pandas as pd
    ids = pd.read_csv(exp.out_dir / "particles.csv").particle.tolist()
    if a.ids:
        ids = [i for i in ids if i in set(a.ids)]
    if a.shard is not None:
        ids = ids[a.shard::a.nshards]
    return ids


def cmd_run(exp, a):
    import pandas as pd
    particles = pd.read_csv(exp.out_dir / "particles.csv")
    band = sim.node_age_band(exp)
    for i in _ids(exp, a):
        if calib.run_path(exp, i).exists() and not a.force:
            print(f"  p{i:04d} done, skipping")
            continue
        t0 = time.time()
        try:
            calib.run_particle(exp, i, particles, band, verbose=a.verbose)
            print(f"  p{i:04d} ok {time.time() - t0:.0f}s", flush=True)
        except Exception:                               # noqa: BLE001 -- keep the shard going
            print(f"  p{i:04d} ERROR\n{traceback.format_exc(limit=3)}", flush=True)


def cmd_score(exp, a):
    s = calib.score(exp)
    if s.empty:
        print("no finished runs")
        return
    cols = ["rank", "particle", "fit_score", "fit_nb_ll", "fit_subtype_ll", "fit_model_total",
            "fit_obs_total", "fit_model_peak_week", "fit_obs_peak_week", "proj_nb_ll",
            "proj_model_total", "proj_obs_total"]
    print(s[[c for c in cols if c in s]].head(a.top).to_string(index=False))
    print(f"-> {exp.out_dir / 'scores.csv'}")


def cmd_status(exp, a):
    import pandas as pd
    p = exp.out_dir / "particles.csv"
    n = len(pd.read_csv(p)) if p.exists() else 0
    done = len(list((exp.out_dir / "runs").glob("p*[0-9].parquet"))) if (exp.out_dir / "runs").exists() else 0
    built = (exp.work_dir / "input.h5").exists()
    print(f"{exp.name}: input.h5 {'ready' if built else 'MISSING (run prepare)'} | "
          f"particles {n} | runs done {done}/{n}")


def cmd_submit(exp, a):
    sb = Path(__file__).with_name("flu_calib.sbatch")
    logs = exp.out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = ["sbatch", f"--array=0-{a.nshards - 1}", f"--job-name=flu-{exp.name}",
           f"--output={logs}/run-%A_%a.out",
           f"--export=ALL,EXP={exp.name},NSHARDS={a.nshards}", str(sb)]
    print("+", " ".join(cmd))
    if not a.dry_run:
        print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "sample", "run", "score", "status", "submit"):
        p = sp.add_parser(name)
        p.add_argument("experiment")
        if name == "prepare":
            p.add_argument("--no-build", action="store_true", help="skip the C++ build")
        if name == "sample":
            p.add_argument("--n", type=int, default=None)
            p.add_argument("--force", action="store_true")
        if name == "run":
            p.add_argument("--ids", type=int, nargs="*")
            p.add_argument("--shard", type=int)
            p.add_argument("--nshards", type=int, default=1)
            p.add_argument("--force", action="store_true")
            p.add_argument("--verbose", action="store_true")
        if name == "score":
            p.add_argument("--top", type=int, default=15)
        if name == "submit":
            p.add_argument("--nshards", type=int, required=True)
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exp = experiment.load(a.experiment)
    {"prepare": cmd_prepare, "sample": cmd_sample, "run": cmd_run, "score": cmd_score,
     "status": cmd_status, "submit": cmd_submit}[a.cmd](exp, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
