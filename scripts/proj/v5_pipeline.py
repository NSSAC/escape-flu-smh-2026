#!/usr/bin/env python
"""prod-v5 (2026-11-06): prodv3g_<ST> (prod-v3's runs with prod-60-v2's guard re-applied) topped up to at least
--target (45) runs per state by resampling the SURVIVING particles, then re-guarding.

    python scripts/proj/v5_pipeline.py make    [ST ...]   # prodv5_<ST> for every (or the given) state with prodv3g_
    python scripts/proj/v5_pipeline.py plan    [ST ...]   # kept / deficit / expected pass rate / draws per state
    python scripts/proj/v5_pipeline.py status  [ST ...]
    python scripts/proj/v5_pipeline.py submit  [ST ...] [--target 45] [pilot options]  # the driver below, as a job
    python scripts/proj/v5_pipeline.py run     [ST ...]   # the driver itself (blocking)

prodv5_<ST> = prodv3_<ST>'s experiment (model v4, hub-matched vaccine, tau x1.16, hosp_prob_scale x1.08) under a new
name. Its restart/ holds per-run symlinks to the v3g KEPT members (all four scenarios; restart/base.csv, never
re-judged) plus real files for the new members. Nothing under proj_prodv3_* or proj_prodv3g_* is written.

Resampling: each round, a state short of the target draws n = ceil(deficit / p x --margin) new members by
systematic resampling (equal weights) of its current survivors (base + new members that passed), so a parameter set
is drawn in proportion to how many of its runs survived. A new member copies the drawn member's row (same parameters,
calibration source and parent; the simulator is unseeded, so it is a new trajectory) with a new id s5000.., a new
run_grouping 5001.. and topup_round = the round. p = the survivors' expected pass rate: the mean over survivors of
their parent's v3 pass probability (peak_ok x size weight over all of that parent's v3 runs; floor 0.5).
Round: B for the pending members (flu_proj.py run --scenarios B --selected, existing runs skipped) -> guard on B
(peak on/after 2026-11-01, size taper 1.5-2x the state record as a Bernoulli keep, draw seeded per member) ->
passers kept, failures dropped from selected.csv. Up to --rounds rounds, then A C D for the new survivors (all of
them, so a state may end above the target), then the analyses (tag prod-v5, --all-deaths, 4 at a time).
restart/v5_draws.csv: every new member (from_member, round, status pending/kept/dropped, guard columns).

A state at or above the target is linked only (no runs), so prodv5_* covers every state and can be submitted as one
set (submission/v3-update/build_v3_update.sh with PREFIX=prodv5 TAG=prod-v5). Rerunning is safe: made experiments
are kept, judged members are not re-judged, finished runs are not redone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py-modules"))
from flu_smh import experiment, proj  # noqa: E402

NB = REPO.parent / "flu-round1-2026-27" / "notebooks"
LOGS = Path(f"/scratch/{os.environ.get('USER', 'user')}/flu-round1-2026-27/sw/drivers")
NAME = "prodv5_{}"
TAG = "prod-v5"
PEAK_AFTER = "2026-11-01"
SEED = 20261107
FIRST_ID, FIRST_RG = 5000, 5001
P_FLOOR = 0.5


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


def states_with(prefix: str) -> list[str]:
    return sorted(p.stem.split("_", 1)[1] for p in experiment.EXPERIMENTS.glob(f"{prefix}_*.json"))


def rdir(exp) -> Path:
    return proj.out_dir(exp) / "restart"


# --------------------------------------------------------------------------------------------------------------
# make

def make(st: str) -> str | None:
    gf = experiment.EXPERIMENTS / f"prodv3g_{st}.json"
    if not gf.exists():
        print(f"{NAME.format(st)}: no prodv3g_{st} yet (its v3 runs are not all in); not made", flush=True)
        return None
    g, v3 = experiment.load(f"prodv3g_{st}"), experiment.load(f"prodv3_{st}")
    name = NAME.format(st)
    f = experiment.EXPERIMENTS / f"{name}.json"
    if not f.exists():
        d = json.loads((experiment.EXPERIMENTS / f"{v3.name}.json").read_text())
        d["name"] = name
        d["notes"] = (f"prod-v5 (2026-11-06): {g.name}'s kept runs + new members resampled from the survivors and "
                      f"re-guarded, to >= the target per state. scripts/proj/v5_pipeline.py. " + d.get("notes", ""))
        d.setdefault("provenance", {})["prodv5_from"] = g.name
        f.write_text(json.dumps(d, indent=1))
    exp = experiment.load(name)
    cal = exp.out_dir
    cal.mkdir(parents=True, exist_ok=True)
    for p in v3.out_dir.iterdir():
        q = cal / p.name
        if not q.exists() and not q.is_symlink():
            q.symlink_to(p.resolve())
    r, gr, vd = rdir(exp), rdir(g), proj.out_dir(v3)
    r.mkdir(parents=True, exist_ok=True)
    for rel in ("members.csv", "restart/screened.csv"):
        if (vd / rel).exists() and not (proj.out_dir(exp) / rel).exists():
            (proj.out_dir(exp) / rel).write_bytes((vd / rel).read_bytes())
    base = pd.read_csv(gr / "selected.csv")
    if not (r / "base.csv").exists():
        base.to_csv(r / "base.csv", index=False)
        base.to_csv(r / "selected.csv", index=False)
    for sid in proj.scenario_ids(v3):
        for m in base.member:
            for ext in (".parquet", ".json"):
                p = proj.member_path(v3, "restart", sid, m).with_suffix(ext)
                q = proj.member_path(exp, "restart", sid, m).with_suffix(ext)
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists() and not q.is_symlink() and p.exists():
                    q.symlink_to(p.resolve())
    home, shome = proj.input_home(exp, "restart"), proj.input_home(v3, "restart")
    home.parent.mkdir(parents=True, exist_ok=True)
    if not home.exists() and not home.is_symlink() and shome.exists():
        home.symlink_to(shome.resolve())
    return name


# --------------------------------------------------------------------------------------------------------------
# state bookkeeping

def draws_of(exp) -> pd.DataFrame:
    f = rdir(exp) / "v5_draws.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["member", "from_member", "round", "status"])


def survivors(exp) -> pd.DataFrame:
    """selected.csv rows of base + new kept (not pending)."""
    sel, d = pd.read_csv(rdir(exp) / "selected.csv"), draws_of(exp)
    pending = set(d.member[d.status == "pending"])
    return sel[~sel.member.isin(pending)]


def parent_pass(st: str) -> pd.Series:
    """Per parent: mean over its v3 runs of peak_ok x size weight (prodv3g's guard table)."""
    gr = rdir(experiment.load(f"prodv3g_{st}"))
    g = pd.read_csv(gr / "guard_v3.csv").merge(pd.read_csv(gr / "selected_unguarded.csv")[["member", "parent"]],
                                               on="member")
    return (g.peak_ok.astype(float) * g.size_weight).groupby(g.parent).mean()


def expected_pass(exp, surv: pd.DataFrame) -> float:
    pp = parent_pass(exp.state)
    return max(P_FLOOR, float(surv.parent.map(pp).fillna(pp.mean()).mean()))


def draw(exp, n: int, rnd: int) -> list[str]:
    """n new members by systematic resampling of the survivors -> selected.csv + v5_draws.csv (pending)."""
    r = rdir(exp)
    sel, d, surv = pd.read_csv(r / "selected.csv"), draws_of(exp), survivors(exp)
    surv = surv.sort_values("member").reset_index(drop=True)
    u0 = np.random.default_rng([SEED, crc(exp.state), rnd]).random()
    idx = np.floor((u0 + np.arange(n)) / n * len(surv)).astype(int)
    used = len(d)
    rows, drec = [], []
    for k, i in enumerate(idx):
        row = surv.iloc[i].copy()
        mid = f"s{FIRST_ID + used + k:04d}"
        drec.append({"member": mid, "from_member": row.member, "round": rnd, "status": "pending"})
        row["member"], row["run_grouping"], row["topup_round"] = mid, FIRST_RG + used + k, rnd
        rows.append(row)
    if "topup_round" not in sel:
        sel["topup_round"] = 0
    pd.concat([sel, pd.DataFrame(rows)], ignore_index=True).to_csv(r / "selected.csv", index=False)
    pd.concat([d, pd.DataFrame(drec)], ignore_index=True).to_csv(r / "v5_draws.csv", index=False)
    return [x["member"] for x in drec]


def judge(exp) -> tuple[int, int]:
    """Guard the pending members whose B run is in: kept / dropped (dropped leave selected.csv)."""
    r, d = rdir(exp), draws_of(exp)
    bsid = proj.scenario_id(exp, "B")
    pend = [m for m in d.member[d.status == "pending"]
            if proj.member_path(exp, "restart", bsid, m).with_suffix(".json").exists()]
    if not pend:
        return 0, 0
    tt = proj.member_totals(exp, "restart", bsid).set_index("member").reindex(pend).reset_index()
    ok = proj.peak_ok(tt, PEAK_AFTER)
    w, _ = proj.guard_weights(exp, tt)
    u = np.array([np.random.default_rng([SEED, crc(exp.state), crc(m)]).random() for m in pend])
    keep = ok & (u < w)
    info = pd.DataFrame({"member": pend, "status": np.where(keep, "kept", "dropped"),
                         "peak_week": pd.to_datetime(tt.proj_model_peak_week).dt.date.astype(str),
                         "total": tt.proj_model_total, "peak": tt.proj_model_peak, "size_weight": w,
                         "peak_ok": ok, "draw": u}).set_index("member")
    d = d.set_index("member")
    for c in info.columns:
        d.loc[info.index, c] = info[c]
    d.reset_index().to_csv(r / "v5_draws.csv", index=False)
    drop = set(info.index[~keep])
    sel = pd.read_csv(r / "selected.csv")
    sel[~sel.member.isin(drop)].to_csv(r / "selected.csv", index=False)
    return int(keep.sum()), len(drop)


def table(states: list[str], target: int, margin: float) -> pd.DataFrame:
    rows = []
    for st in states:
        f = experiment.EXPERIMENTS / f"{NAME.format(st)}.json"
        if not f.exists():
            continue
        e = experiment.load(NAME.format(st))
        surv, d = survivors(e), draws_of(e)
        p = expected_pass(e, surv)
        deficit = max(0, target - len(surv))
        rows.append({"state": st, "base": len(pd.read_csv(rdir(e) / "base.csv")), "survivors": len(surv),
                     "deficit": deficit, "p_pass": round(p, 2),
                     "next_draws": math.ceil(deficit / p * margin) if deficit else 0,
                     "drawn": len(d), "kept_new": int((d.status == "kept").sum()),
                     "dropped": int((d.status == "dropped").sum()), "pending": int((d.status == "pending").sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------
# driver

def flu_proj(names: list[str], scenarios: list[str], a) -> int:
    cmd = [sys.executable, "-u", str(REPO / "scripts/proj/flu_proj.py"), "run", *names, "--mode", "restart",
           "--selected", "--scenarios", *scenarios, "--partition", a.partition, "--jobs", str(a.jobs),
           *(["--qos", a.qos] if a.qos else []),
           *(["--extra-partition", a.extra_partition, "--extra-jobs", str(a.extra_jobs)] if a.extra_partition else []),
           *(["--extra-qos", a.extra_qos] if a.extra_qos else [])]
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def analyse(states: list[str], jobs: int) -> None:
    def one(st):
        cmd = [sys.executable, "-u", str(NB / "run_analyses.py"), NAME.format(st), "--tag", TAG, "--us", "never",
               "--upscale-to", "300", "--all-deaths", "--no-index"]
        print("+", shlex.join(cmd), flush=True)
        return subprocess.run(cmd, cwd=NB).returncode
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, states))
    subprocess.run([sys.executable, "-u", str(NB / "run_analyses.py"), "--tag", TAG, "--index-only"], cwd=NB)
    every = states_with("prodv5")
    if len(every) == len(states_with("prodv3")):
        subprocess.run([sys.executable, "-u", str(NB / "run_analyses.py"), *[NAME.format(s) for s in every],
                        "--tag", TAG, "--states", "US", "--us", "always", "--upscale-to", "300", "--all-deaths"],
                       cwd=NB)


def cmd_run(a, states: list[str]) -> int:
    made = [st for st in states if make(st)]
    print(time.strftime("%F %T"), f"prodv5: {len(made)} states, target {a.target}", flush=True)
    for rnd in range(1, a.rounds + 1):
        for st in made:
            e = experiment.load(NAME.format(st))
            judge(e)                                         # runs left from an earlier (interrupted) driver
            d, surv = draws_of(e), survivors(e)
            deficit = a.target - len(surv)
            if deficit > 0 and not (d.status == "pending").any():
                n = math.ceil(deficit / expected_pass(e, surv) * a.margin)
                draw(e, n, int(d["round"].max()) + 1 if len(d) else 1)
        t = table(made, a.target, a.margin)
        print(f"\n== round {rnd}\n{t.to_string(index=False)}", flush=True)
        todo = list(t.state[t.pending > 0])
        if not todo:
            break
        flu_proj([NAME.format(s) for s in todo], ["B"], a)
        for st in todo:
            k, x = judge(experiment.load(NAME.format(st)))
            print(f"  {st}: {k} kept, {x} dropped", flush=True)
    t = table(made, a.target, a.margin)
    print(f"\n== after B\n{t.to_string(index=False)}", flush=True)
    grown = list(t.state[t.kept_new > 0])
    if grown:
        flu_proj([NAME.format(s) for s in grown], ["A", "C", "D"], a)
    for st in made:                                          # pending members never got a B run: leave them out
        e = experiment.load(NAME.format(st))
        d = draws_of(e)
        pend = set(d.member[d.status == "pending"])
        if pend:
            sel = pd.read_csv(rdir(e) / "selected.csv")
            sel[~sel.member.isin(pend)].to_csv(rdir(e) / "selected.csv", index=False)
            d.loc[d.member.isin(pend), "status"] = "no_run"
            d.to_csv(rdir(e) / "v5_draws.csv", index=False)
    if not a.no_analysis:
        analyse(made, a.analysis_jobs)
    print(f"\n{time.strftime('%F %T')} done\n{table(made, a.target, a.margin).to_string(index=False)}", flush=True)
    return 0


def cmd_submit(a, argv: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    inner = argv[argv.index("submit") + 1:]
    script = (f". /etc/profile; module load gcc/14.2.0; eval \"$($HOME/conda/bin/conda shell.bash hook)\"; "
              f"conda activate escape-sw; cd {REPO}; exec python -u scripts/proj/v5_pipeline.py run "
              + shlex.join(inner))
    cmd = ["sbatch", "--job-name=prodv5", f"--account={a.account}", f"--partition={a.driver_partition}",
           "--nodes=1", "--ntasks-per-node=1", "--cpus-per-task=8", "--mem=96G", f"--time={a.driver_time}",
           f"--output={LOGS}/%x-%j.out", *([f"--dependency={a.dependency}"] if a.dependency else []),
           "--wrap", script]
    print("+", shlex.join(cmd))
    if a.dry_run:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["make", "plan", "status", "run", "submit"])
    ap.add_argument("states", nargs="*", help="default: every state with prodv3g_")
    ap.add_argument("--target", type=int, default=45, help="minimum runs per state")
    ap.add_argument("--margin", type=float, default=1.2, help="draws = deficit / pass rate x margin")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--partition", default="bii")
    ap.add_argument("--qos", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--extra-partition", default="bii-largemem")
    ap.add_argument("--extra-qos", default=None)
    ap.add_argument("--extra-jobs", type=int, default=12)
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--analysis-jobs", type=int, default=4)
    ap.add_argument("--account", default="bii_nssac")
    ap.add_argument("--driver-partition", default="bii-largemem")
    ap.add_argument("--driver-time", default="2-00:00:00")
    ap.add_argument("--dependency", default=None, help="submit: e.g. afterany:22531981")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    states = [s.upper() for s in a.states] or states_with("prodv3g")
    if a.cmd == "make":
        for st in states:
            make(st)
        print(table(states, a.target, a.margin).to_string(index=False))
        return 0
    if a.cmd in ("plan", "status"):
        if a.cmd == "plan":
            for st in states:
                make(st)
        t = table(states, a.target, a.margin)
        print(t.to_string(index=False))
        print(f"\n{len(t)} states; {int((t.deficit > 0).sum())} short of {a.target}; next draws {int(t.next_draws.sum())}")
        return 0
    if a.cmd == "submit":
        return cmd_submit(a, argv)
    return cmd_run(a, states)


if __name__ == "__main__":
    sys.exit(main())
