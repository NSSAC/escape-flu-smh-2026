"""Build, prepare, run, and extract models/flu-r1-2026-27.esl for an Experiment.

    prepare(exp)                       schedule -> render -> build -> input.h5 (once per experiment)
    run(exp, configs, out_h5)          one simulation
    extract(exp, out_h5)               -> weekly long table: date (week-ending Sat), subtype, age_group,
                                          infections, admissions, deaths
    stats(exp, out_h5)                 per-tick statistics (incl. susceptible_mass_<sub> for S0)

The HDF5 output is streamed tick by tick (flu_<sub>/transitions/tick_N/{node_index, state} and
.../interventions/tick_N for seeded infections), so a two-year VA run (~1 GB) never sits in memory.
State codes follow the ESL enum order: S V E I A H D R. node_index is the row in input.h5's node table,
which is the person file sorted by pid.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import schedule
from .experiment import SUBTYPES, Experiment

STATE = {"S": 0, "V": 1, "E": 2, "I": 3, "A": 4, "H": 5, "D": 6, "R": 7}
#: age bands for outputs: NHSN's
AGE_EDGES = [0, 5, 18, 50, 65, 200]
AGE_LABELS = ["0-4", "5-17", "18-49", "50-64", "65-130"]


def schedule_path(exp: Experiment) -> Path:
    return exp.work_dir / "vax_schedule.csv"


def input_path(exp: Experiment) -> Path:
    return exp.work_dir / "input.h5"


def make_simulator(exp: Experiment):
    from escape_abm.utils import OpenMPSimulator
    sched = pd.read_csv(schedule_path(exp))
    rows = schedule.esl_rows(sched) or [{"tick": 0, "new_vax_count": 0, "min_age": 0, "max_age": 0,
                                         "vax_campaign": 0, "vax_formulation_tick": 0,
                                         "propensity_max": 0.0}]     # no campaign (scenario D): one empty row
    return OpenMPSimulator(exp.model_path, exp.work_dir, vax_schedule=rows)


def prepare(exp: Experiment, build: bool = True) -> None:
    exp.work_dir.mkdir(parents=True, exist_ok=True)
    sched = schedule.build(exp)
    sched.to_csv(schedule_path(exp), index=False)
    sim = make_simulator(exp)
    if build:
        sim.prepare_build()
        sim.build()
    if not input_path(exp).exists():
        sim.prepare_input(str(exp.person_path), str(exp.network_path), input_path(exp))


def run(exp: Experiment, configs: dict, out_h5: Path, verbose: bool = False) -> Path:
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    sim = make_simulator(exp)
    sim.simulate(input_path(exp), out_h5, num_ticks=exp.num_ticks,
                 configs={k: float(v) for k, v in configs.items()}, verbose=verbose)
    return out_h5


def population(exp: Experiment) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(exp.person_path).metadata.num_rows


def node_age_band(exp: Experiment) -> np.ndarray:
    """Age-band index (into AGE_LABELS) for every node, in input.h5 node order (= sorted pid)."""
    p = pd.read_parquet(exp.person_path, columns=["pid", "age"]).sort_values("pid")
    return (np.digitize(p.age.to_numpy(), AGE_EDGES[1:-1])).astype(np.int8)


def extract(exp: Experiment, out_h5: Path, band: np.ndarray | None = None) -> pd.DataFrame:
    band = band if band is not None else node_age_band(exp)
    nb = len(AGE_LABELS)
    T = exp.num_ticks
    rows = []
    with h5py.File(out_h5, "r") as f:
        for sub in SUBTYPES:
            g = f[f"flu_{sub}"]
            inf = np.zeros((T, nb), dtype=np.int64)
            adm = np.zeros((T, nb), dtype=np.int64)
            dth = np.zeros((T, nb), dtype=np.int64)
            for t in range(T):
                key = f"tick_{t}"
                for grp in ("transitions", "interventions"):
                    if key not in g[grp]:
                        continue
                    st = g[grp][key]["state"][()]
                    if st.size == 0:
                        continue
                    b = band[g[grp][key]["node_index"][()]]
                    inf[t] += np.bincount(b[st == STATE["E"]], minlength=nb)
                    if grp == "transitions":
                        adm[t] += np.bincount(b[st == STATE["H"]], minlength=nb)
                        dth[t] += np.bincount(b[st == STATE["D"]], minlength=nb)
            for j, a in enumerate(AGE_LABELS):
                rows.append(pd.DataFrame({"tick": np.arange(T), "subtype": sub, "age_group": a,
                                          "infections": inf[:, j], "admissions": adm[:, j],
                                          "deaths": dth[:, j]}))
    d = pd.concat(rows, ignore_index=True)
    d["date"] = exp.t0 + pd.to_timedelta(d.tick, unit="D")
    d["week_end"] = d.date + pd.to_timedelta((5 - d.date.dt.dayofweek) % 7, unit="D")
    w = (d.groupby(["week_end", "subtype", "age_group"], as_index=False)
         [["infections", "admissions", "deaths"]].sum().rename(columns={"week_end": "date"}))
    return w


def stats(exp: Experiment, out_h5: Path) -> pd.DataFrame:
    """Per-tick statistics (new_hosp_count_*, susceptible_mass_*, ...), via escape_abm's reader."""
    return make_simulator(exp).extract_statistics(out_h5).to_pandas()
