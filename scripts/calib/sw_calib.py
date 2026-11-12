#!/usr/bin/env python
"""Calibrate several experiments at once on pilot jobs (slurm-workflows + slurm-workflows-optimize).

    module load gcc/14.2.0 && conda activate escape-sw          # in escape-flu-smh
    python scripts/calib/sw_calib.py explore  bt2023_VA bt2023_WA bt2023_MA --points 64 --jobs 24
    python scripts/calib/sw_calib.py optimize bt2023_VA bt2023_WA bt2023_MA --parallelism 8 --jobs 24
    python scripts/calib/sw_calib.py submit   explore bt2023_VA bt2023_WA bt2023_MA --points 64 --jobs 24
    swtop <address printed by the driver>                         # watch it (any shell, same env)

explore    ExploreSpaceSobolQMC: a Sobol design of --points (power of 2) per experiment over its priors
           (flu_smh.calib.space), every point of every experiment in one pool of pilot jobs.
optimize   OptimizeSpaceBotorch: batch Bayesian optimization (--parallelism points per round per
           experiment), modelled on every results file the experiment already has (explore + earlier searches).
submit     run either of the above as a long-lived driver job (sw_calib.sbatch), so it outlives this shell.

The driver hosts the ds-service server (on ib0) and submits pilot jobs to the bii partition: one 40-core
node per job, one worker per job (is_batch_worker), which runs one simulation at a time. Each worker
imports flu_smh from this repo's py-modules (flu_smh/sw.py holds the objective).

Outputs, per experiment (flu-round1-2026-27/calib_<name>/):
  runs/sw<hash>.parquet (+ .stats.parquet, .json)   one per parameter set, same format as flu_calib.py runs
  sw/explore-s<seed>-n<points>.pkl.gz            the exploration's points/values/outputs (results-file format)
  sw/search-<stamp>.pkl.gz                       each optimize run's points
  scores.csv                                     `flu_calib.py score` is rerun at the end
  report/index.html (+ PNGs, CSVs)               flu-round1-2026-27/notebooks/calib_report.py, run at the end
With several experiments, also the summary page, named after its calibration iteration
(flu-round1-2026-27/calib_reports/it04-reduced.html; experiments/iterations.json), and calib_reports/index.html.
Pilot-job scripts and logs: /scratch/$USER/flu-round1-2026-27/sw/<stamp>-<command>/.

Resuming: a run's id is a hash of its parameters, so a point already simulated is scored from disk and
never re-run. After an interruption, rerun the same explore command (same seed): only missing points
simulate. optimize always starts from every sw/*.pkl.gz file of the experiment.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py-modules"))
from flu_smh import calib, experiment, sim, sw  # noqa: E402

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


def sbatch_args(a, cpus=40, mem="0") -> list[str]:
    qos = f" --qos={a.qos}" if a.qos else ""
    return [f"--account={a.account} --partition={a.partition}{qos}",
            f"--nodes=1 --ntasks-per-node=1 --cpus-per-task={cpus} --mem={mem}",
            f"--time={a.pilot_time}"]


def results_files(exp) -> list[Path]:
    return sorted((exp.out_dir / "sw").glob("*.pkl.gz"))


def save_split(results: dict, exps: list, stem: str) -> None:
    """One results file per experiment, in its own out_dir/sw/ (the results-file format, so either class
    can load it back)."""
    for exp in exps:
        r = results[exp.name]
        p = exp.out_dir / "sw" / f"{stem}.pkl.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wb") as f:
            pickle.dump({exp.name: {"points": r.points, "values": r.values, "outputs": r.outputs}}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{exp.name}: {len(r.values)} points -> {p}")


def prepare(exps, a, ex) -> None:
    """Schedule, build and input.h5 for each experiment, as tasks on the pilot jobs (the driver is a 2-core,
    16 GB job: one big state's input.h5 is beyond it). Experiments whose input.h5 exists only rebuild."""
    from slurm_workflows import RaiseOnError
    if a.no_prepare:
        for exp in exps:
            assert sim.input_path(exp).exists(), f"{exp.name}: not prepared ({sim.input_path(exp)})"
        return
    t0 = time.time()
    tasks = [ex.submit(["sim"], sw.prepare_task, exp.name, task_priority=priority(exp)) for exp in exps]
    ex.wait(tasks, desc="prepare", raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED)
    print(f"{len(exps)} experiments prepared in {time.time() - t0:.0f}s", flush=True)


def priority(exp) -> float:
    """Bigger states first, so the long runs start early and the short ones fill in at the end."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(exp.person_path).metadata.num_rows / 1e6


def score_all(exps) -> None:
    for exp in exps:
        s = calib.score(exp)
        if s.empty:
            continue
        b = s.iloc[0]
        print(f"{exp.name}: {len(s)} runs scored -> {exp.out_dir / 'scores.csv'}; best {b.particle} "
              f"fit_score {b.fit_score:.1f}, fit total {b.fit_model_total:.0f} vs obs {b.fit_obs_total:.0f}, "
              f"proj total {b.get('proj_model_total', float('nan')):.0f} vs obs "
              f"{b.get('proj_obs_total', float('nan')):.0f}")


def make_reports(exps, a) -> None:
    """The calibration report (calib_report.py, in the flu-round1-2026-27 notebooks). A separate process,
    so a plotting problem never costs a finished calibration."""
    script = experiment.FLU_ROUND / "notebooks" / "calib_report.py"
    if a.no_report or not script.exists():
        return
    cmd = [sys.executable, str(script), *[e.name for e in exps], "--no-score"]
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd)


def run(a) -> int:
    from ds_service_client import DsServiceServer
    from slurm_workflows import SlurmPilotExecutor

    exps = [experiment.load(n) for n in a.experiments]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = SW_SCRATCH / f"{stamp}-{a.cmd}"
    n_eval = (a.points if a.cmd == "explore" else a.parallelism) * len(exps)
    jobs = min(a.jobs, n_eval)
    failed = False
    with DsServiceServer(interface=a.interface) as ds:
        ds.wait_until_ready()
        print(f"ds-service at {ds.address}  (watch: swtop {ds.address})", flush=True)
        with SlurmPilotExecutor(f"flu-{a.cmd}", ds.address, work_dir=work) as ex:
            ex.define_job_group("sim", sbatch_args(a), SETUP_SCRIPT, is_batch_worker=True,
                                python_paths=[REPO / "py-modules"], add_cwd_to_python_path=False)
            ex.scale_jobs("sim", jobs)
            print(f"{jobs} pilot jobs on {a.partition} for {n_eval} evaluations", flush=True)
            prepare(exps, a, ex)
            if a.cmd == "explore":
                from slurm_workflows_optimize import ExplorationStudy, ExploreSpaceSobolQMC
                studies = [ExplorationStudy(exp.name, sw.space(exp), sw.objective, "sim",
                                            num_exploration_points=a.points,
                                            seed=a.seed if a.seed is not None else exp.seed,
                                            extra_objective_kwargs={"experiment": exp.name},
                                            priority=priority(exp)) for exp in exps]
                job = ExploreSpaceSobolQMC(studies, ex)
                stem = lambda st: f"explore-s{st.seed}-n{st.num_exploration_points}"  # noqa: E731
            else:
                from slurm_workflows_optimize import OptimizationStudy, OptimizeSpaceBotorch
                ex.define_job_group("gp", sbatch_args(a, cpus=8, mem="32G"), SETUP_SCRIPT, is_batch_worker=True,
                                    python_paths=[REPO / "py-modules"], add_cwd_to_python_path=False)
                ex.scale_jobs("gp", 1)
                studies = [OptimizationStudy(exp.name, sw.space(exp), sw.objective, "sim", "gp",
                                             search_parallelism=a.parallelism,
                                             min_search_rounds=a.min_rounds, max_search_rounds=a.max_rounds,
                                             patience=a.patience, min_improvement=a.min_improvement,
                                             extra_objective_kwargs={"experiment": exp.name},
                                             priority=priority(exp)) for exp in exps]
                files = [f for exp in exps for f in results_files(exp)]
                assert files, "optimize needs results files: run explore first"
                job = OptimizeSpaceBotorch(studies, ex, files)
                stem = lambda st: f"search-{stamp}"  # noqa: E731
            try:
                job.run()
            except RuntimeError as e:
                failed = True
                print(f"!! {e}\n   (points that came back are saved; rerun to fill in the rest; "
                      f"worker logs in {work})", flush=True)
            for st in job.studies:
                save_split(job.results, [e for e in exps if e.name == st.name], stem(st))
    score_all(exps)
    make_reports(exps, a)
    return 1 if failed else 0


def submit(a, argv) -> int:
    sb = Path(__file__).with_name("sw_calib.sbatch")
    logs = SW_SCRATCH / "drivers"
    logs.mkdir(parents=True, exist_ok=True)
    inner = argv[argv.index(a.cmd, argv.index("submit") + 1):]
    cmd = ["sbatch", f"--job-name=sw-{a.cmd}", f"--account={a.account}", f"--partition={a.partition}",
           f"--time={a.driver_time}", f"--output={logs}/%x-%j.out",
           *([f"--qos={a.qos}"] if a.qos else []), str(sb), *inner]
    print("+", shlex.join(cmd))
    if a.dry_run:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode == 0:
        print(f"driver log: {logs}/sw-{a.cmd}-<jobid>.out (it prints the swtop address)")
    return r.returncode


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("experiments", nargs="+")
        p.add_argument("--jobs", type=int, default=8, help="pilot jobs (40-core nodes); capped at the evaluations")
        p.add_argument("--account", default="bii_nssac")
        p.add_argument("--partition", default="bii")
        p.add_argument("--qos", default=None)
        p.add_argument("--pilot-time", default="2-00:00:00", help="pilot-job time limit (a run cut off by "
                       "it stays Running on the server: rerun the command to finish it)")
        p.add_argument("--interface", default="ib0")
        p.add_argument("--no-prepare", action="store_true", help="skip schedule/build (input.h5 must exist)")
        p.add_argument("--no-report", action="store_true", help="skip calib_report.py at the end")

    e = sp.add_parser("explore")
    common(e)
    e.add_argument("--points", type=int, default=16, help="Sobol points per experiment (floored to 2^k)")
    e.add_argument("--seed", type=int, default=None, help="default: the experiment's seed")
    o = sp.add_parser("optimize")
    common(o)
    o.add_argument("--parallelism", type=int, default=8, help="points per round per experiment")
    o.add_argument("--min-rounds", type=int, default=3)
    o.add_argument("--max-rounds", type=int, default=12)
    o.add_argument("--patience", type=int, default=3)
    o.add_argument("--min-improvement", type=float, default=0.01,
                   help="fractional improvement of log(-fit_score) a round needs to count")
    s = sp.add_parser("submit", help="run `explore ...` or `optimize ...` as a driver job")
    s.add_argument("--driver-time", default="3-00:00:00")
    s.add_argument("--dry-run", action="store_true")
    ssp = s.add_subparsers(dest="inner", required=True)
    for name, p in (("explore", e), ("optimize", o)):
        ssp.add_parser(name, parents=[p], add_help=False, conflict_handler="resolve")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parser().parse_args(argv)
    if a.cmd == "submit":
        a.cmd = a.inner
        return submit(a, argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
