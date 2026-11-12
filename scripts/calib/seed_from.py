#!/usr/bin/env python
"""Seed an experiment's search with another experiment's runs, rescored under the new experiment's objective.

    python scripts/calib/seed_from.py bt2023l_VA:bt2023m_VA bt2024l_WA:bt2024m_WA ...

For a new iteration that changes only the objective (likelihood), not the model, the priors or the
calendar: every simulated point of <src> is a valid evaluation of <dst> too. This copies <src>'s runs into
<dst>'s runs/ (a run id is a hash of its parameters, so sw.objective finds them and scores from disk) and
writes <dst>/sw/seed-<src>.pkl.gz with every point of <src>'s results files, valued by <dst>'s objective.
`sw_calib.py optimize <dst>` then starts from those points, as it would from an explore.
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
from flu_smh import experiment, sw  # noqa: E402

#: experiment fields that must match for the source runs to stand for the destination's
SAME = ("state", "person_file", "anchor", "end", "fit_start", "fit_end", "proj_start", "proj_end",
        "season_break", "campaigns", "fixed", "priors", "draws", "model", "next_season", "share_map")


def seed(src_name: str, dst_name: str) -> Path:
    src, dst = experiment.load(src_name), experiment.load(dst_name)
    diff = [k for k in SAME if getattr(src, k) != getattr(dst, k)]
    assert not diff, f"{src_name} -> {dst_name}: runs are not interchangeable, {diff} differ"
    (dst.out_dir / "runs").mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for f in (src.out_dir / "runs").iterdir():
        t = dst.out_dir / "runs" / f.name
        if not t.exists():
            shutil.copy2(f, t)
            n_copied += 1
    points, values, outputs = [], [], []
    for rf in sorted((src.out_dir / "sw").glob("*.pkl.gz")):
        with gzip.open(rf, "rb") as fh:
            r = pickle.load(fh)[src_name]
        for p in r["points"]:
            out = sw.objective(dst_name, **p)
            assert out["reused"], f"{dst_name}: {out['run_id']} was simulated, not found on disk"
            points.append(p)
            values.append(out["objective"])
            outputs.append(out)
    p = dst.out_dir / "sw" / f"seed-{src_name}.pkl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb") as fh:
        pickle.dump({dst_name: {"points": points, "values": values, "outputs": outputs}}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{src_name} -> {dst_name}: {n_copied} run files copied, {len(points)} points rescored -> {p}", flush=True)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", metavar="SRC:DST")
    a = ap.parse_args(argv)
    for pair in a.pairs:
        seed(*pair.split(":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
