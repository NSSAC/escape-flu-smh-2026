#!/usr/bin/env python
"""Hub projections on pilot jobs: calibrated production experiments -> paired trajectories, scenarios A-D.

    module load gcc/14.2.0 && conda activate escape-sw          # in escape-flu-smh
    python scripts/proj/flu_proj.py members prod2025k_VA prod2025k_WA --n 300 --top 20
    python scripts/proj/flu_proj.py run     prod2025k_VA prod2025k_WA --mode restart --jobs 48 --qos bii-half
    python scripts/proj/flu_proj.py submit  run prod2025k_VA prod2025k_WA --mode restart --jobs 48 --qos bii-half
    python scripts/proj/flu_proj.py status  prod2025k_VA prod2025k_WA --mode restart
    python scripts/proj/flu_proj.py select  prod2025k_VA --mode restart --n 300    # optional, after a B pass

members  proj_<name>/members.csv from the calibration's scores.csv (flu_smh.proj.members): the best `--top` runs'
         searched parameters, each member with its own next-season draws; run_grouping = member index + 1.
run      prepare every (experiment, scenario) build on the workers (the first scenario of each experiment
         first, since in restart mode it builds the shared input.h5), then every missing (scenario, member)
         run, biggest states first. A finished run is never redone, so rerunning the same command resumes.
         --scenarios B --oversample-pass then `select`, then --scenarios A C D --selected: the optional
         history-weighted selection (flu_smh.proj.select).
status   finished runs per experiment and scenario, and the run-time distribution.
score    round 2 of the two-stage design (backtests): each member's projection-window likelihood, totals, peaks
         -> proj_<name>/<mode>/<sid>_scores.csv; notebooks/next_season_report.py reads them.

TWO-STAGE ROUND 2 (backtests, it05):
    flu_proj.py members bt2023l_VA ... --design sobol --top 5 --per-fit 32     # 160 members per experiment
    flu_proj.py submit run bt2023l_VA ... --jobs 48 --qos bii-half             # restart, scenario B only
    flu_proj.py score bt2023l_VA ...; python notebooks/next_season_report.py it05
submit   `run ...` as a long-lived driver job (scripts/calib/sw_calib.sbatch's pattern), so it outlives this shell.

PRODUCTION 2026-27 (it08, age re-anchored; the season began before the origin, so the members are
screened on its early weeks):
    flu_proj.py members prod2025o_* --design sobol --top 20 --per-fit 64          # 1280 candidates per state
    flu_proj.py submit run prod2025o_* --mode screen --scenarios B --jobs 95       # runs cut at the origin
    flu_proj.py select prod2025o_* --mode screen --weights early --n 900 --ess 300 # -> restart/screened.csv
    flu_proj.py submit run prod2025o_* --mode restart --selected --scenarios B --jobs 95   # 900 x B, full season
    flu_proj.py select prod2025o_* --mode restart --weights guard --n 300          # -> restart/selected.csv
    flu_proj.py submit run prod2025o_* --mode restart --selected --scenarios A C D --jobs 95
    python notebooks/smh_submission.py prod2025o_* --selected
  screen   restart mode cut a week after the origin (~1/5 of a full run): scored on the projected season's
           weeks before the hub's fitting cutoff (proj.early_window: 2026-08-09 .. 2026-10-03)
  select   weights ~ exp(early_score / T), T set so the effective sample size is --ess; 300 draws with
           replacement -> new members s0000.. (a repeated parameter set is a new trajectory: no seed).
           --ess is judged on the backtests: next_season_report.py's "selection" section.
  guard    the full B runs of the screened members against the state's past NHSN seasons (proj.guard_weights):
           weight 1 up to 1.5x the largest season since 2022-23, cos^2 down to 0 at 2x it (the lower of the total's
           and the peak's weights); n kept by PPS sampling (inclusion ~ weight) among weight > 0, so the screen
           selection oversamples (3x: 900 for 300; preview-v3 WA had 34 of 90 under 2x).

Outputs: flu-round1-2026-27/proj_<name>/<mode>/<scenario_id>/m####.parquet (+ .json with configs and S0).
The hub file: flu-round1-2026-27/notebooks/smh_submission.py.

Capacity: one simulation per pilot job; --cpus/--mem size the job (40 / whole node by default; 20 + 180G packs two
per bii node). --extra-partition adds a second job group (e.g. bii-largemem, whose QOS has no per-user CPU cap)
serving the same queue.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py-modules"))
from flu_smh import experiment, proj, sim  # noqa: E402

ENV = os.environ.get("FLU_SW_ENV", "escape-sw")
SW_SCRATCH = experiment.SCRATCH.parent / "sw"

SETUP_SCRIPT = f"""
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate {ENV}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-40}}
export FLU_ROUND={shlex.quote(str(experiment.FLU_ROUND))}
export FLU_SCRATCH={shlex.quote(str(experiment.SCRATCH))}
"""


def sbatch_args(a, partition: str, qos: str | None) -> list[str]:
    q = f" --qos={qos}" if qos else ""
    return [f"--account={a.account} --partition={partition}{q}",
            f"--nodes=1 --ntasks-per-node=1 --cpus-per-task={a.cpus} --mem={a.mem}",
            f"--time={a.pilot_time}"]


def size(exp) -> float:
    return sim.population(exp) / 1e6


def cmd_members(a) -> int:
    for name in a.experiments:
        exp = experiment.load(name)
        m = proj.members(exp, n=a.n, top=a.top, weights=a.weights, temperature=a.temperature, design=a.design,
                         per_fit=a.per_fit)
        print(f"{name}: {len(m)} members from {m.source.nunique()} calibration runs "
              f"(fit_score {m.source_fit_score.max():.1f} .. {m.source_fit_score.min():.1f}) -> "
              f"{proj.out_dir(exp) / 'members.csv'}")
    return 0


def plan(a):
    exps = [experiment.load(n) for n in a.experiments]
    sids = {e.name: [proj.scenario_id(e, s) for s in (a.scenarios or proj.scenario_ids(e))] for e in exps}
    todo = {e.name: proj.missing(e, a.mode, sids[e.name], proj.load_members(e, a.selected, a.mode)) for e in exps}
    return exps, sids, todo


def cmd_run(a) -> int:
    from ds_service_client import DsServiceServer
    from slurm_workflows import RaiseOnError, SlurmPilotExecutor

    exps, sids, todo = plan(a)
    n = sum(len(v) for v in todo.values())
    for e in exps:
        print(f"{e.name} [{a.mode}]: {len(todo[e.name])} runs to do over {', '.join(sids[e.name])}", flush=True)
    if n == 0:
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = SW_SCRATCH / f"{stamp}-proj"
    jobs = min(a.jobs, n)
    with DsServiceServer(interface=a.interface) as ds:
        ds.wait_until_ready()
        print(f"ds-service at {ds.address}  (watch: swtop {ds.address})", flush=True)
        with SlurmPilotExecutor("flu-proj", ds.address, work_dir=work) as ex:
            queues = ["sim"]
            ex.define_job_group("sim", sbatch_args(a, a.partition, a.qos), SETUP_SCRIPT, is_batch_worker=True,
                                python_paths=[REPO / "py-modules"], add_cwd_to_python_path=False)
            ex.scale_jobs("sim", jobs)
            if a.extra_partition and a.extra_jobs:
                ex.define_job_group("sim2", sbatch_args(a, a.extra_partition, a.extra_qos), SETUP_SCRIPT,
                                    is_batch_worker=True, python_paths=[REPO / "py-modules"],
                                    add_cwd_to_python_path=False)
                ex.scale_jobs("sim2", min(a.extra_jobs, n))
                queues.append("sim2")
            print(f"{jobs}{f' + {a.extra_jobs}' if len(queues) > 1 else ''} pilot jobs for {n} runs", flush=True)
            preps, runs = [], []
            for e in exps:
                pri = size(e)
                first = ex.submit(queues, proj.prepare_task, e.name, sids[e.name][0], a.mode,
                                  task_priority=1e6 + pri)
                prep = {sids[e.name][0]: first}
                for sid in sids[e.name][1:]:
                    prep[sid] = ex.submit(queues, proj.prepare_task, e.name, sid, a.mode, task_parents=[first],
                                          task_priority=1e6 + pri)
                preps += list(prep.values())
                for sid, m in todo[e.name]:
                    runs.append(ex.submit(queues, proj.run_task, e.name, sid, m, a.mode,
                                          task_parents=[prep[sid]], task_priority=pri))
            failed = False
            try:
                ex.wait(preps, desc="prepare", raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED)
                ex.wait(runs, desc="runs", unit="run", raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED)
            except RuntimeError as err:
                failed = True
                print(f"!! {err}\n   (finished runs are on disk; rerun the command to fill in the rest; "
                      f"worker logs in {work})", flush=True)
    cmd_status(a)
    return 1 if failed else 0


def cmd_status(a) -> int:
    import json

    import numpy as np
    for name in a.experiments:
        e = experiment.load(name)
        mem = proj.load_members(e, a.selected, a.mode) if (proj.out_dir(e) / "members.csv").exists() else None
        parts = []
        secs = []
        for sid in proj.scenario_ids(e):
            js = list((proj.out_dir(e) / a.mode / sid).glob("[ms][0-9]*.json"))
            parts.append(f"{sid.split('-')[0]} {len(js)}")
            secs += [json.loads(j.read_text())["run_seconds"] for j in js[:200]]
        tot = f"/{len(mem)}" if mem is not None else " (no members.csv)"
        rt = (f"; run min p10/50/90 {np.percentile(np.array(secs) / 60, [10, 50, 90]).round(1).tolist()}"
              if secs else "")
        print(f"{name} [{a.mode}]: {', '.join(parts)}{tot}{rt}")
    return 0


def cmd_score(a) -> int:
    for name in a.experiments:
        e = experiment.load(name)
        for sid in (a.scenarios or ["B"]):
            sc = proj.score_members(e, a.mode, sid)
            if sc.empty:
                print(f"{name}: no finished {sid} members")
                continue
            b = sc.iloc[0]
            print(f"{name} [{a.mode}/{sid}]: {len(sc)} members scored -> {proj.out_dir(e) / a.mode}; best {b.member} "
                  f"(fit rank {b.source_rank}) proj_nb_ll {b.proj_nb_ll:.1f}, total {b.proj_model_total:.0f} vs obs "
                  f"{b.proj_obs_total:.0f}, peak {b.proj_model_peak_week} vs {b.proj_obs_peak_week}")
    return 0


def cmd_select(a) -> int:
    for name in a.experiments:
        e = experiment.load(name)
        proj.select(e, a.mode, n=a.n, weights=a.weights, target_ess=a.ess)
    return 0


def cmd_submit(a, argv) -> int:
    sb = REPO / "scripts" / "proj" / "flu_proj.sbatch"
    logs = SW_SCRATCH / "drivers"
    logs.mkdir(parents=True, exist_ok=True)
    inner = argv[argv.index("run", argv.index("submit") + 1):]
    cmd = ["sbatch", "--job-name=flu-proj", f"--account={a.account}", f"--partition={a.partition}",
           f"--time={a.driver_time}", f"--output={logs}/%x-%j.out", *([f"--qos={a.qos}"] if a.qos else []),
           str(sb), *inner]
    print("+", shlex.join(cmd))
    if a.dry_run:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("experiments", nargs="+")
        p.add_argument("--mode", default="restart", choices=proj.MODES)
        p.add_argument("--selected", action="store_true", help="run the members of <mode>/selected.csv")

    m = sp.add_parser("members")
    m.add_argument("experiments", nargs="+")
    m.add_argument("--n", type=int, default=300)
    m.add_argument("--top", type=int, default=20)
    m.add_argument("--weights", default="uniform", choices=["uniform", "fit"])
    m.add_argument("--temperature", type=float, default=1.0, help="fit weights ~ exp(fit_score / T)")
    m.add_argument("--design", default="random", choices=["random", "sobol"],
                   help="sobol: round 2 of the two-stage design, --top fits x --per-fit Sobol points (n ignored)")
    m.add_argument("--per-fit", type=int, default=32)

    r = sp.add_parser("run")
    common(r)
    r.add_argument("--scenarios", nargs="*", help="A B C D (default: all)")
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--cpus", type=int, default=40)
    r.add_argument("--mem", default="0")
    r.add_argument("--account", default="bii_nssac")
    r.add_argument("--partition", default="bii")
    r.add_argument("--qos", default=None)
    r.add_argument("--extra-partition", default=None, help="a second job group, e.g. bii-largemem")
    r.add_argument("--extra-qos", default=None)
    r.add_argument("--extra-jobs", type=int, default=0)
    r.add_argument("--pilot-time", default="2-00:00:00")
    r.add_argument("--interface", default="ib0")

    s = sp.add_parser("status")
    common(s)
    sc = sp.add_parser("score", help="round 2: score the members on the projection window (backtests)")
    common(sc)
    sc.add_argument("--scenarios", nargs="*", help="default: B")
    se = sp.add_parser("select")
    common(se)
    se.add_argument("--n", type=int, default=300)
    se.add_argument("--weights", default="history", choices=["history", "uniform", "early", "guard"])
    se.add_argument("--ess", type=float, default=None, help="weights=early: the tempered weights' effective "
                    "sample size (default n / 2)")

    sb = sp.add_parser("submit", help="run `run ...` as a driver job")
    sb.add_argument("--driver-time", default="3-00:00:00")
    sb.add_argument("--dry-run", action="store_true")
    ssp = sb.add_subparsers(dest="inner", required=True)
    ssp.add_parser("run", parents=[r], add_help=False, conflict_handler="resolve")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parser().parse_args(argv)
    if a.cmd == "submit":
        return cmd_submit(a, argv)
    return {"members": cmd_members, "run": cmd_run, "status": cmd_status, "score": cmd_score,
            "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
