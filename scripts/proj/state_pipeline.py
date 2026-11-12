#!/usr/bin/env python
"""The production preview for many states in ONE driver and one pool of pilot jobs, each state moving through
its own chain of steps as soon as its previous step finishes, and analysed the moment its last run lands: a state
is done when analysis/<tag>/<ST>/index.html appears (and pipeline_status.csv, the overview page and `status`
say "done").

    module load gcc/14.2.0 && conda activate escape-sw          # in escape-flu-smh
    python scripts/proj/state_pipeline.py submit run prod2025o_* --tag preview-v3 --jobs 54 --qos bii-half
    python scripts/proj/state_pipeline.py status --tag preview-v3          # any shell, while it runs
    python scripts/proj/state_pipeline.py run ...                          # the same as `submit run`, here

One state's chain (every step resumes from disk, so resubmitting the same command continues where it stopped):
  prepare   schedule, build and input.h5 of the round-1 experiment                      pilot task
  explore   round 1: the Sobol design over the 2025-26 priors (--points)                 pilot tasks
            -> calib_<name>/sw/explore-s<seed>-n<points>.pkl.gz, scores.csv (calib_report.py queued)
  members   top --top fits x --per-fit next-season Sobol draws -> members.csv            driver
  screen    scenario B, cut a week after the origin                                       pilot tasks
  early     --screen-n members at ESS --ess on the early-season weeks -> restart/screened.csv   driver
  B         the full scenario-B pass of the screened members                              pilot tasks
  topup     while the guard's weight mass over the B runs is under --margin x --keep-n: more screened members
            drawn from the same early-season weights (proj.topup), and their B runs     driver + pilot tasks
            (at most --max-topups rounds and --max-b x --keep-n B runs)
  guard     --keep-n kept against the past seasons -> restart/selection.csv, pass_rates.csv   driver
  ACD       scenarios A C D for the kept members                                          pilot tasks
  analysis  run_analyses.py <name> --tag <tag> --us never                                  driver, one at a time
When every chain has ended: the US (the sum of the states, PARTIAL if one failed) and the overview page.

Sizing: the guard keeps --keep-n by PPS on its weights, which is proportional only while the weights sum to at
least --keep-n; the top-up keeps that sum at --margin x --keep-n. The top-up draws from the early-season-weighted
screen candidates, not from the guard's survivors (that would weigh the guard twice and push out the high-burden
parameter sets); the guard then runs once over every B run. --pass-rates <earlier tag>/pass_rates.csv sizes each
state's first screen selection from its pass rate, so most states need no top-up.

Aim high, settle lower (--keep-n 60 --min-keep 45): top-ups only until the guard mass reaches --margin x
--min-keep; the guard then keeps up to --keep-n of the passing runs (fewer only where fewer pass).

Peak timing (--peak-after 2026-11-01): the guard also gives weight 0 to a B run peaking (week ending) before the
date, a hard cut (no state peaked before November in NHSN 2022-26). With a finished selection, every state whose
kept set holds such a run is reopened as with --grow (only the failing members are replaced); use --max-topups 2
where prod-60 already topped up once (the round count is per state, over all runs).

Growing a finished preview (--grow --keep-n 60): a state whose guard kept fewer than --keep-n reopens its guard
selection (proj.grow_open): the kept members stay kept, the B step tops up toward --margin x --keep-n, the guard
draws the rest, and only the new members run A C D. --upscale-to N passes N trajectories per state (severity
replicates, flu_smh/upscale.py) to the analyses.

Steps whose output exists are not redone: members.csv, restart/screened.csv (early) and restart/selection.csv
(guard) are kept, so a state already through a step is never re-selected (delete the file to redo it). The run
steps only run the missing (scenario, member) runs, and an explore whose results file is complete is skipped, so
states already finished go straight to their analysis.

Order: the pilot jobs take a later step before an earlier one (a state in flight finishes before the next one
starts) and, within a step, the bigger state first (--order size; --order given keeps the command line's order):
the long runs start early and states finish one after another rather than all at the end. A step whose runs
partly failed goes on with the runs that finished (logged with "!!"); a failed prepare or an explore with no
point back ends the state's chain ("failed" in the status).

Status: flu-round1-2026-27/analysis/<tag>/pipeline_status.csv (state, step, runs left, started, done), rewritten
at every step; the analyses' logs in /scratch/$USER/flu-round1-2026-27/sw/drivers/<tag>-analysis/<ST>.log.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py-modules"))
from flu_smh import calib, experiment, proj, sim, sw  # noqa: E402

ENV = os.environ.get("FLU_SW_ENV", "escape-sw")
SW_SCRATCH = experiment.SCRATCH.parent / "sw"
NOTEBOOKS = experiment.FLU_ROUND / "notebooks"
ANALYSIS = experiment.FLU_ROUND / "analysis"

SETUP_SCRIPT = f"""
module load gcc/14.2.0
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate {ENV}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-40}}
export FLU_ROUND={shlex.quote(str(experiment.FLU_ROUND))}
export FLU_SCRATCH={shlex.quote(str(experiment.SCRATCH))}
"""

STEPS = ["prepare", "explore", "members", "screen", "early", "B", "topup", "guard", "ACD", "analysis", "done"]
# a later step outranks an earlier one on the pilots; the prepares (minutes, and everything waits on them) first
RANK = {"explore": 1, "screen": 2, "B": 3, "ACD": 4, "prepare": 10}


class Dropped:
    """The output of a task the server will never run (canceled, unknown, no pilot job left)."""

    def __init__(self, why: str):
        self.error = why


@dataclass
class Chain:
    exp: experiment.Experiment
    order: float
    step: str = "prepare"
    tasks: list = field(default_factory=list)
    note: str = ""
    started: str = ""
    finished: str = ""
    failed: bool = False
    design: list = field(default_factory=list)
    label: str = ""                  # the experiment's name when two chains share a state (test arms)

    @property
    def name(self) -> str:
        return self.exp.name

    @property
    def code(self) -> str:
        return self.label or self.exp.state


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"{stamp()} {msg}", flush=True)


def sbatch_args(a, partition: str, qos: str | None) -> list[str]:
    q = f" --qos={qos}" if qos else ""
    return [f"--account={a.account} --partition={partition}{q}",
            f"--nodes=1 --ntasks-per-node=1 --cpus-per-task={a.cpus} --mem={a.mem}",
            f"--time={a.pilot_time}"]


def explore_file(exp, a) -> Path:
    return exp.out_dir / "sw" / f"explore-s{exp.seed}-n{a.points}.pkl.gz"


def explore_complete(exp, a) -> bool:
    f = explore_file(exp, a)
    if not f.exists():
        return False
    with gzip.open(f, "rb") as fh:
        return len(pickle.load(fh)[exp.name]["values"]) >= a.points


class Pipeline:
    def __init__(self, a, ex):
        self.a, self.ex = a, ex
        self.queues = ["sim"] + (["sim2"] if a.extra_partition and a.extra_jobs else [])
        exps = [experiment.load(n) for n in a.experiments]
        size = {e.name: sim.population(e) / 1e6 for e in exps}
        if a.order == "size":
            exps.sort(key=lambda e: -size[e.name])
        # the state term stays under 1 so it only breaks ties within a step
        self.chains = [Chain(e, 1 - k / (len(exps) + 1)) for k, e in enumerate(exps)]
        states = [e.state for e in exps]
        for c in self.chains:                       # arms of one state: each analysed under <tag>/<experiment>
            if states.count(c.exp.state) > 1:
                c.label = c.exp.name
        self.owner: dict[str, Chain] = {}
        self.local: list[tuple[str, list[str], Path, Chain | None]] = []   # (what, argv, log, chain)
        self.running: tuple[subprocess.Popen, str, Chain | None] | None = None
        self.truth_fetched = False
        self.status_file = ANALYSIS / a.tag / "pipeline_status.csv"
        self.rates = pd.read_csv(a.pass_rates).set_index("state") if a.pass_rates else None
        self.logs = SW_SCRATCH / "drivers" / f"{a.tag}-analysis"
        self.logs.mkdir(parents=True, exist_ok=True)

    # ---- tasks -------------------------------------------------------------------------------------------
    def submit(self, c: Chain, step: str, fn, *args, parents=None, label: str = "", **kw):
        t = self.ex.submit(self.queues, fn, *args, task_parents=parents or None,
                           task_priority=RANK[step] * 100 + c.order, **kw)
        self.ex.set_task_name(t, f"{c.code}-{step}{f'-{label}' if label else ''}")
        self.owner[t.task_id] = c
        c.tasks.append(t)
        return t

    @staticmethod
    def pending(c: Chain) -> list:
        from slurm_workflows.slurm_pilot_executor import NoOutput
        return [t for t in c.tasks if t.output is NoOutput]

    @staticmethod
    def errors(c: Chain) -> list:
        from slurm_workflows import RemoteExecutionError
        return [t for t in c.tasks if isinstance(t.output, (RemoteExecutionError, Dropped))]

    # ---- the chain ---------------------------------------------------------------------------------------
    def enter(self, c: Chain, step: str, note: str = "") -> None:
        c.step, c.note, c.tasks = step, note, []
        log(f"{c.code}: {step}{f' ({note})' if note else ''}")
        self.write_status()

    def fail(self, c: Chain, why: str) -> None:
        c.failed, c.note = True, why
        log(f"!! {c.code}: FAILED at {c.step}: {why}")
        self.write_status()

    def advance(self, c: Chain) -> None:
        """Move one chain on until it waits for pilot tasks or a local job, or ends."""
        while not c.failed and c.step != "done" and not self.pending(c):
            if c.step == "analysis":
                return                                    # the local queue moves it to "done"
            bad = self.errors(c)
            if bad:
                log(f"!! {c.code} {c.step}: {len(bad)} of {len(c.tasks)} tasks failed, e.g. "
                    f"{str(bad[0].output.error).strip().splitlines()[-1][:300]}")
            try:
                self.step(c, bad)
            except Exception as err:                      # one state's problem never stops the others
                import traceback
                traceback.print_exc()
                self.fail(c, f"{type(err).__name__}: {err}")

    def step(self, c: Chain, bad: list) -> None:
        """Finish c.step (its tasks are all back) and start the next one."""
        a, e = self.a, c.exp
        if c.step == "prepare" and not c.tasks:                       # the chain's start
            c.started = stamp()
            if explore_complete(e, a):
                self.enter(c, "explore", "complete on disk")
                return self.after_explore(c, ran=False)
            self.enter(c, "prepare")
            self.submit(c, "prepare", sw.prepare_task, e.name)
            return
        if c.step == "prepare":
            if bad:
                return self.fail(c, "prepare failed (see the error above)")
            return self.start_explore(c)
        if c.step == "explore":
            return self.after_explore(c, ran=True)
        if c.step in ("screen", "B", "ACD") and len(bad) == len(c.tasks) and c.tasks:
            return self.fail(c, f"every {c.step} task failed")
        if c.step == "screen":
            if not (proj.out_dir(e) / "restart" / "screened.csv").exists():
                n = self.first_n(c)
                self.enter(c, "early", f"{n} members")
                proj.select(e, "screen", n=n, weights="early", target_ess=a.ess)
            if (a.grow or a.peak_after) and proj.grow_open(e, a.keep_n, a.peak_after) is not None:
                log(f"{c.code}: reopening the guard selection (keep {a.keep_n}"
                    f"{', peaks from ' + a.peak_after if a.peak_after else ''})")
            return self.start_runs(c, "B", "restart", ["B"], selected=True)
        if c.step == "B":
            if not (proj.out_dir(e) / "restart" / "selection.csv").exists():
                m = proj.guard_mass(e, a.peak_after)
                goal = a.min_keep or a.keep_n                 # top up for min_keep; the guard still keeps up to keep_n
                need, cap = a.margin * goal, a.max_b * a.keep_n
                add = (proj.topup_size(e, goal, a.margin, max_runs=cap, peak_after=a.peak_after)
                       if m["topup_round"] < a.max_topups else 0)
                if add > 0:
                    self.enter(c, "topup", f"round {m['topup_round'] + 1}: guard mass {m['mass']:.1f} of "
                                           f"{need:.1f} over {m['runs']} B runs (pass rate {m['pass_rate']:.2f}): +{add}")
                    proj.topup(e, add)
                    return self.start_runs(c, "B", "restart", ["B"], selected=True)
                if m["mass"] < a.keep_n:
                    log(f"!! {c.code}: guard mass {m['mass']:.1f} < {a.keep_n} after {m['runs']} B runs "
                        f"({m['topup_round']} top-ups, cap {cap}): {min(m['passing'], a.keep_n)} kept "
                        f"({m['passing']} passing), not fully proportional to the guard weight")
                self.enter(c, "guard", f"mass {m['mass']:.1f}, pass rate {m['pass_rate']:.2f}")
                proj.select(e, "restart", n=a.keep_n, weights="guard", keep=proj.grow_kept(e),
                            peak_after=a.peak_after)
                self.record_pass_rate(c, m)
            if a.no_acd:
                return self.enter(c, "ACD", "skipped (--no-acd)")
            return self.start_runs(c, "ACD", "restart", ["A", "C", "D"], selected=True)
        if c.step == "ACD":
            self.enter(c, "analysis", "queued")
            argv = [str(NOTEBOOKS / "run_analyses.py"), e.name, "--tag", self.tag_of(c), "--us", "never",
                    *self.upscale_args()]
            if not self.truth_fetched:
                argv.append("--refresh-truth")
                self.truth_fetched = True
            self.local.insert(sum(1 for w in self.local if w[0] == "analysis"),     # analyses before reports
                              ("analysis", argv, self.logs / f"{c.code}.log", c))
            return

    def upscale_args(self) -> list[str]:
        return ["--upscale-to", str(self.a.upscale_to)] if self.a.upscale_to else []

    def tag_of(self, c: Chain) -> str:
        return f"{self.a.tag}/{c.label}" if c.label else self.a.tag

    def first_n(self, c: Chain) -> int:
        """The first screen selection's size: --screen-n, or, for a state in --pass-rates, enough for the guard
        mass to reach margin x keep_n at that pass rate (at least that mass, at most max_b x keep_n)."""
        a = self.a
        if a.pass_rates is None or c.code not in self.rates.index:
            return a.screen_n
        need = a.margin * a.keep_n
        return int(min(max(np.ceil(need / max(self.rates.loc[c.code, "pass_rate"], 0.02)), need), a.max_b * a.keep_n))

    def record_pass_rate(self, c: Chain, m: dict) -> None:
        """analysis/<tag>/pass_rates.csv: the guard's pass rate per state (--pass-rates for a later, larger run)."""
        f = ANALYSIS / self.a.tag / "pass_rates.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["state"])
        row = pd.DataFrame([{"state": c.code, "experiment": c.name, **m, "keep_n": self.a.keep_n}])
        pd.concat([old[old.state != c.code], row], ignore_index=True).sort_values("state").to_csv(f, index=False)

    def start_explore(self, c: Chain) -> None:
        from slurm_workflows_optimize import ExplorationStudy, ExploreSpaceSobolQMC
        e = c.exp
        study = ExplorationStudy(e.name, sw.space(e), sw.objective, "sim", num_exploration_points=self.a.points,
                                 seed=e.seed, extra_objective_kwargs={"experiment": e.name})
        design = ExploreSpaceSobolQMC([study], self.ex).design(e.name)   # the design only: sw_calib.py's points
        self.enter(c, "explore", f"{len(design)} points")
        c.design = design
        for i, p in enumerate(design):
            self.submit(c, "explore", sw.objective, label=f"{i:03d}", experiment=e.name, **p)

    def after_explore(self, c: Chain, ran: bool) -> None:
        e = c.exp
        if ran:
            ok = [(p, t.output) for p, t in zip(c.design, c.tasks) if isinstance(t.output, dict)]
            if not ok:
                return self.fail(c, "no explore point came back")
            f = explore_file(e, self.a)
            f.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(f, "wb") as fh:
                pickle.dump({e.name: {"points": [p for p, _ in ok], "values": [o["objective"] for _, o in ok],
                                      "outputs": [dict(o) for _, o in ok]}}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            s = calib.score(e)
            log(f"{c.code}: {len(ok)} explore points -> {f.name}; best fit_score {s.fit_score.iloc[0]:.1f}")
            if not self.a.no_calib_report:
                self.local.append(("report", [str(NOTEBOOKS / "calib_report.py"), e.name, "--no-score"],
                                   self.logs / f"{c.code}-calib-report.log", None))
        elif not (e.out_dir / "scores.csv").exists():
            calib.score(e)
        if not (proj.out_dir(e) / "members.csv").exists():
            self.enter(c, "members")
            # production["top"] / ["per_fit"] override the command line per experiment (prodv4_*: each state's
            # original design, 2026-11-05)
            m = proj.members(e, top=int(e.production.get("top", self.a.top)), design="sobol",
                             per_fit=int(e.production.get("per_fit", self.a.per_fit)))
            log(f"{c.code}: {len(m)} candidates from {m.source.nunique()} fits")
        self.start_runs(c, "screen", "screen", ["B"], selected=False)

    def start_runs(self, c: Chain, step: str, mode: str, scen: list[str], selected: bool) -> None:
        e = c.exp
        sids = [proj.scenario_id(e, s) for s in scen]
        todo = proj.missing(e, mode, sids, proj.load_members(e, selected, mode))
        self.enter(c, step, f"{len(todo)} runs")
        if not todo:
            return                                        # advance() moves on: nothing pending
        prep = {sids[0]: self.submit(c, "prepare", proj.prepare_task, e.name, sids[0], mode, label=step)}
        for sid in sids[1:]:
            prep[sid] = self.submit(c, "prepare", proj.prepare_task, e.name, sid, mode, parents=[prep[sids[0]]],
                                    label=step)
        for sid, m in todo:
            self.submit(c, step, proj.run_task, e.name, sid, m, mode, parents=[prep[sid]],
                        label=f"{sid.split('-')[0]}-{m['member']}")

    # ---- the local queue: analyses and calibration reports, one at a time, in the driver -----------------
    def poll_local(self) -> bool:
        """Start/finish local jobs; True when one finished (so the caller re-checks the chains)."""
        changed = False
        if self.running and self.running[0].poll() is not None:
            p, what, c = self.running
            if c is not None:
                c.finished = stamp()
                if p.returncode:
                    c.note = f"analysis exited {p.returncode}: {self.logs / (c.code + '.log')}"
                    log(f"!! {c.code}: {c.note}")
                else:
                    c.note = ""
                    log(f"=== {c.code} DONE -> {ANALYSIS / self.tag_of(c) / c.exp.state / 'index.html'}")
                c.step = "done"
                self.write_status()
            elif p.returncode:
                log(f"!! {what} exited {p.returncode}")
            self.running, changed = None, True
        if self.running is None and self.local:
            what, argv, logf, c = self.local.pop(0)
            if c is not None:
                c.note = "running"
                self.write_status()
            fh = open(logf, "w")
            self.running = (subprocess.Popen([sys.executable, "-u", *argv], cwd=NOTEBOOKS, stdout=fh,
                                             stderr=subprocess.STDOUT), f"{what} {Path(argv[1]).name}", c)
        return changed

    # ---- status ------------------------------------------------------------------------------------------
    def write_status(self) -> None:
        rows = [{"state": c.code, "experiment": c.name,
                 "step": "failed" if c.failed else c.step,
                 "tasks_left": len(self.pending(c)), "tasks": len(c.tasks), "note": c.note,
                 "started": c.started, "done": c.finished if c.step == "done" else ""} for c in self.chains]
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows)
        if self.status_file.exists():          # a later run on the same tag (a few states): keep the others' rows
            try:
                old = pd.read_csv(self.status_file, keep_default_na=False)
                new = pd.concat([old[~old.state.isin(new.state)], new], ignore_index=True)
            except (pd.errors.EmptyDataError, KeyError):
                pass
        tmp = self.status_file.with_suffix(".tmp")
        new.to_csv(tmp, index=False)
        tmp.replace(self.status_file)

    # ---- the loop ----------------------------------------------------------------------------------------
    def run(self) -> int:
        stuck = 0
        last_status = 0.0
        while True:
            for c in self.chains:
                self.advance(c)
            self.poll_local()
            live = [c for c in self.chains if not c.failed and c.step != "done"]
            if not live and self.running is None and not self.local:
                break
            if time.time() - last_status > 300:
                self.write_status()
                last_status = time.time()
            pending = [t for c in live for t in self.pending(c)]
            if not pending:
                time.sleep(15)
                continue
            try:
                for t in self.ex.as_completed(pending, desc="pipeline", unit="task"):
                    c = self.owner[t.task_id]
                    if not self.pending(c) or self.poll_local():
                        break
                stuck = 0
            except RuntimeError as err:
                msg = str(err)
                for tid in re.findall(r"Task (\S+) (?:was canceled|is unknown)", msg):
                    t = next((t for t in pending if t.task_id == tid), None)
                    if t is not None:
                        t.output = Dropped(msg)
                if "no live pilot job" in msg or "no worker started" in msg:
                    stuck += 1
                    log(f"!! {msg}")
                    if stuck > 3 or not self.top_up():   # the stranded tasks stay queued on the server
                        log("!! no pilot jobs left to run the pending tasks: stopping (resubmit to continue)")
                        self.write_status()
                        return 1
        self.us()
        self.write_status()
        failed = [c.code for c in self.chains if c.failed]
        log(f"=== all chains ended: {len(self.chains) - len(failed)} done"
            f"{f', FAILED {failed} (resubmit to retry)' if failed else ''}")
        return 1 if failed else 0

    def top_up(self) -> bool:
        """Forget pilot jobs that have left the queue and submit replacements (a node failure, a time limit)."""
        from slurm_workflows.slurm_utils import get_running_jobids
        alive = get_running_jobids()
        n = 0
        for q, want in (("sim", self.a.jobs), ("sim2", self.a.extra_jobs)):
            g = self.ex.groups.get(q)
            if g is None:
                continue
            for k, j in list(g.jobs.items()):
                if j.job_id not in alive:
                    del g.jobs[k]
                    n += 1
            self.ex.scale_jobs(q, want)
        log(f"replaced {n} pilot jobs that had ended")
        return n > 0

    def us(self) -> None:
        done = [c.name for c in self.chains if c.step == "done" and not c.label]
        if len(done) < 2 or self.a.no_us:
            return
        argv = [*done, "--tag", self.a.tag, "--states", "US", "--us", "always", *self.upscale_args()]
        log(f"US: {len(done)} states -> analysis/{self.a.tag}/US")
        with open(self.logs / "US.log", "w") as fh:
            r = subprocess.run([sys.executable, "-u", str(NOTEBOOKS / "run_analyses.py"), *argv], cwd=NOTEBOOKS,
                               stdout=fh, stderr=subprocess.STDOUT)
        log(f"{'!! ' if r.returncode else ''}US analysis exited {r.returncode} ({self.logs / 'US.log'})")


def cmd_run(a) -> int:
    from ds_service_client import DsServiceServer
    from slurm_workflows import SlurmPilotExecutor

    work = SW_SCRATCH / f"{time.strftime('%Y%m%d-%H%M%S')}-pipeline"
    with DsServiceServer(interface=a.interface) as ds:
        ds.wait_until_ready()
        log(f"ds-service at {ds.address}  (watch: swtop {ds.address})")
        with SlurmPilotExecutor("flu-pipe", ds.address, work_dir=work) as ex:
            ex.define_job_group("sim", sbatch_args(a, a.partition, a.qos), SETUP_SCRIPT, is_batch_worker=True,
                                python_paths=[REPO / "py-modules"], add_cwd_to_python_path=False)
            ex.scale_jobs("sim", a.jobs)
            if a.extra_partition and a.extra_jobs:
                ex.define_job_group("sim2", sbatch_args(a, a.extra_partition, a.extra_qos), SETUP_SCRIPT,
                                    is_batch_worker=True, python_paths=[REPO / "py-modules"],
                                    add_cwd_to_python_path=False)
                ex.scale_jobs("sim2", a.extra_jobs)
            pl = Pipeline(a, ex)
            log(f"{len(pl.chains)} states, {a.jobs}{f' + {a.extra_jobs}' if a.extra_jobs else ''} pilot jobs; "
                f"status: {pl.status_file}; worker logs: {work}")
            return pl.run()


def cmd_status(a) -> int:
    f = ANALYSIS / a.tag / "pipeline_status.csv"
    if not f.exists():
        print(f"no {f}")
        return 1
    s = pd.read_csv(f, keep_default_na=False)
    print(f"{f}  (written {time.strftime('%F %T', time.localtime(f.stat().st_mtime))})")
    print(s.step.value_counts().reindex(STEPS + ["failed"]).dropna().astype(int).to_string())
    print(s.to_string(index=False))
    return 0


def cmd_submit(a, argv) -> int:
    sb = REPO / "scripts" / "proj" / "state_pipeline.sbatch"
    logs = SW_SCRATCH / "drivers"
    logs.mkdir(parents=True, exist_ok=True)
    inner = argv[argv.index("run", argv.index("submit") + 1):]
    cmd = ["sbatch", "--job-name=flu-pipeline", f"--account={a.account}", f"--partition={a.partition}",
           f"--time={a.driver_time}", f"--output={logs}/%x-%j.out", *([f"--qos={a.qos}"] if a.qos else []),
           str(sb), *inner]
    print("+", shlex.join(cmd))
    if a.dry_run:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode == 0:
        print(f"driver log: {logs}/flu-pipeline-<jobid>.out; progress: state_pipeline.py status --tag {a.tag}")
    return r.returncode


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("experiments", nargs="+")
    r.add_argument("--tag", required=True, help="analysis/<tag>/")
    r.add_argument("--order", default="size", choices=["size", "given"])
    r.add_argument("--points", type=int, default=64, help="round-1 Sobol points (a power of 2)")
    r.add_argument("--top", type=int, default=8)
    r.add_argument("--per-fit", type=int, default=16)
    r.add_argument("--screen-n", type=int, default=90)
    r.add_argument("--ess", type=float, default=30)
    r.add_argument("--keep-n", type=int, default=30)
    r.add_argument("--margin", type=float, default=1.25, help="guard mass wanted: margin x keep-n")
    r.add_argument("--min-keep", type=int, default=None,
                   help="top up only for margin x min-keep (the guard still keeps up to --keep-n of the passing "
                        "runs): aim for keep-n, settle for min-keep")
    r.add_argument("--max-topups", type=int, default=2, help="top-up rounds per state (0: none)")
    r.add_argument("--max-b", type=float, default=6, help="B runs per state at most max-b x keep-n")
    r.add_argument("--pass-rates", default=None, help="a pass_rates.csv (an earlier run's analysis/<tag>/): "
                   "each listed state's first screen selection sized for margin x keep-n at its pass rate")
    r.add_argument("--grow", action="store_true",
                   help="reopen finished guard selections with fewer than --keep-n kept: keep those, top up B runs, "
                        "select the rest (proj.grow_open), run A C D for the new members only")
    r.add_argument("--upscale-to", type=int, default=None, metavar="N",
                   help="analyses with N trajectories per state (run_analyses.py --upscale-to: severity replicates)")
    r.add_argument("--no-calib-report", action="store_true")
    r.add_argument("--peak-after", metavar="DATE",
                   help="peak-timing guard: weight 0 for a B run whose all-age weekly peak (week ending) is before "
                        "DATE (e.g. 2026-11-01); a finished selection keeping such a run is reopened like --grow: "
                        "the passing kept members stay, the rest are re-drawn (top-ups from the screen weights)")
    r.add_argument("--no-acd", action="store_true", help="stop after the guard (scenario B only: tests)")
    r.add_argument("--no-us", action="store_true")
    r.add_argument("--jobs", type=int, default=48)
    r.add_argument("--cpus", type=int, default=40)
    r.add_argument("--mem", default="0")
    r.add_argument("--account", default="bii_nssac")
    r.add_argument("--partition", default="bii")
    r.add_argument("--qos", default=None)
    r.add_argument("--extra-partition", default=None, help="a second job group, e.g. bii-largemem")
    r.add_argument("--extra-qos", default=None)
    r.add_argument("--extra-jobs", type=int, default=0)
    r.add_argument("--pilot-time", default="7-00:00:00")
    r.add_argument("--interface", default="ib0")
    s = sp.add_parser("status")
    s.add_argument("--tag", required=True)
    sb = sp.add_parser("submit", help="`run ...` as a driver job")
    sb.add_argument("--driver-time", default="7-00:00:00")
    sb.add_argument("--dry-run", action="store_true")
    ssp = sb.add_subparsers(dest="inner", required=True)
    ssp.add_parser("run", parents=[r], add_help=False, conflict_handler="resolve")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parser().parse_args(argv)
    if a.cmd == "submit":
        return cmd_submit(a, argv)
    return {"run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
