"""Run the SPOK-24 model (models/spok24.esl) for the SMH cryptic round.

Paths and conventions in one place (Round 20's drivers hard-coded them per script):

  person file      cryptic-round/data/person/<LOC>-person.parquet   (scripts/build_person_parquet.py)
  network          EpiHiperSynPop v2.4.0 usa_<st>/<st>_contact_network.csv
  importations     cryptic-round/data/importation/<LOC>/<library>/events.parquet (notebooks/importation.py)
  build dir        <scratch>/<LOC>/<library>/        one build per (location, schedule library)
  input.h5         <scratch>/<LOC>/input.h5          shared by every library of a location

A schedule library is compiled into the model, so pick a schedule per run with the
`import_schedule_id` config -- no rebuild. A new library means one new build (a few minutes).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

REPO = Path(__file__).resolve().parents[2]
CRYPTIC = Path("/project/bii_nssac/biocomplexity/SMH/cryptic-round")
SIM_FILE = REPO / "models" / "spok24.esl"
PERSON_FILE_FMT = str(CRYPTIC / "data/person/{loc}-person.parquet")
NETWORK_FILE_FMT = "/project/bii_nssac/EpiHiperSynPop/v2.4.0/from-alt-project/usa_{st}/{st}_contact_network.csv"
LIBRARY_DIR_FMT = str(CRYPTIC / "data/importation/{loc}/{library}")
SCRATCH = Path(os.environ.get("SPOK24_SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}/cryptic-round"))

SIM_START_DATE = pl.date(2024, 9, 22)      # tick 0 (a Sunday)
CALIB_END_DATE = pl.date(2025, 1, 18)      # tick 118
NUM_CALIB_TICKS = 119
MAX_TICK = 1000                            # schedule key = schedule_id * MAX_TICK + tick (must match the ESL)


def import_events(loc: str, library: str) -> list[dict]:
    ev = pl.read_parquet(Path(LIBRARY_DIR_FMT.format(loc=loc, library=library)) / "events.parquet")
    assert ev["tick"].max() < MAX_TICK
    return [dict(key=int(s) * MAX_TICK + int(t), n=int(n))
            for s, t, n in ev.select("schedule_id", "tick", "n").iter_rows()]


def make_simulator(loc: str, library: str):
    from escape_abm.utils import OpenMPSimulator
    return OpenMPSimulator(SIM_FILE, SCRATCH / loc / library, import_events=import_events(loc, library))


def input_file(loc: str) -> Path:
    return SCRATCH / loc / "input.h5"


def prepare(loc: str, library: str, build: bool = True) -> None:
    """Build the simulator for (loc, library) and create the location's input.h5 if missing."""
    sim = make_simulator(loc, library)
    if build:
        sim.prepare_build()
        sim.build()
    inp = input_file(loc)
    if not inp.exists():
        inp.parent.mkdir(parents=True, exist_ok=True)
        sim.prepare_input(PERSON_FILE_FMT.format(loc=loc), NETWORK_FILE_FMT.format(st=loc.lower()), inp)


def run(loc: str, library: str, configs: dict[str, Any], out_dir: str | Path,
        num_ticks: int = NUM_CALIB_TICKS, verbose: bool = False) -> Path:
    sim = make_simulator(loc, library)
    out = Path(out_dir) / f"output-{uuid4()}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    sim.simulate(input_file(loc), out, num_ticks=num_ticks, configs=configs, verbose=verbose)
    return out


def weekly_statistics(loc: str, library: str, output_file: str | Path) -> pl.DataFrame:
    """Weekly admissions and importations (weeks ending Saturday) from the run's statistics."""
    sim = make_simulator(loc, library)
    return (sim.extract_statistics(output_file)
            .with_columns((SIM_START_DATE + pl.duration(days="tick")).alias("date"))
            .with_columns((pl.col("date") + pl.duration(days=(13 - pl.col("date").dt.weekday()) % 7))
                          .alias("week_end_date"))
            .group_by("week_end_date")
            .agg(pl.sum("new_hosp_count"), pl.sum("new_import_count"), pl.sum("imports_scheduled"))
            .sort("week_end_date"))
