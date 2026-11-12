"""Smoke test for models/spok24.esl: build, prepare input, run a few schedules, print weekly output.

    module load gcc/14.2.0
    conda activate escape            # escape-abm, polars
    python scripts/test_spok24.py VA v1 0 17 293
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py-modules"))
from escape_covid19_smh import spok24  # noqa: E402

loc = sys.argv[1] if len(sys.argv) > 1 else "VA"
library = sys.argv[2] if len(sys.argv) > 2 else "v1"
schedules = [int(s) for s in sys.argv[3:]] or [0]

t0 = time.time()
spok24.prepare(loc, library)
print(f"prepared {loc}/{library} in {time.time() - t0:.0f}s", flush=True)

for sid in schedules:
    t1 = time.time()
    out = spok24.run(loc, library, {"import_schedule_id": sid}, spok24.SCRATCH / loc / "test" / f"sched{sid}")
    wk = spok24.weekly_statistics(loc, library, out)
    print(f"\n=== schedule {sid}: {time.time() - t1:.0f}s -> {out}")
    print(wk.to_pandas().to_string(index=False), flush=True)
