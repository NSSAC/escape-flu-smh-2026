#!/usr/bin/env python
"""prodv3_<ST>: a state's prod-60-v2 production experiment re-run under the v4 vaccine with transmission and
hospitalization rescaled instead of recalibrated (2026-11-05; tag prod-v3, a test before a v4 recalibration).

    python scripts/proj/make_prodv3.py VA MA WA FL            # experiments/prodv3_<ST>.json + the dirs below
    python scripts/proj/flu_proj.py submit run prodv3_FL prodv3_VA prodv3_WA prodv3_MA --mode restart --selected \\
        --qos bii-half --jobs 50 --extra-partition bii-largemem --extra-jobs 40

What changes (flu-round1-2026-27/notebooks/ve_runs.py, ve_analysis.py; docs/flu.md "Realized VE"):
  model   flu-r1-2026-27-v4.esl (v3 + vax_escape_per_day, vax_symp_ve; v2/v3 behaviour by default)
  vaccine vax_escape_per_day 1 (no escape: the hub's 50% holds whatever the match), vax_infection_ve 0.43,
          vax_symp_ve 0.46, vax_hosp_ve 0 -> realized (matched cohort, VA B) ~23% infection, ~49% symptomatic,
          ~52% hospitalization, vs ~10% / 10% / 33% under v2
  tau     tau_mult 1.16 (calib.configs_for: baseline_tau and tau_seasonality_scale x 1.16): restores the 2025-26
          attack rate the stronger vaccine removed (VA sweep: 1.15-1.18 by calibration run)
  hosp    hosp_prob_scale x 1.08: the admissions per infection lost to the vaccine at that multiplier (1.04-1.13)
Everything else is the base experiment's: the same 60 selected members (restart/selected.csv, same parameters and
run_grouping), not re-screened or re-guarded, so the early-season fit and the peak-timing guard are NOT re-applied
(the comparison shows how far they drift). The restart's tick-0 S0 targets stay the old calibration runs' (the
ltimm fractions are re-solved under the new vaccine history, informed_priors.s0_history).

RECALIBRATION (--recalib -> prodv4_<ST>, tag prod-v4): the same model, vaccine, tau_mult and hosp_prob_scale, but
the whole chain redone by state_pipeline.py like the base: round-1 explore of the 2025-26 season (the base's
priors and seed, so the same Sobol points, now under v4 with tau x 1.16), members (production.top / per_fit = the
base's design: 8 x 16 for prod2025o, x 32 for seedx2, x 64 for seedx3_FL), screen, early selection, B, guard (with
the Nov-1 peak guard), A C D, analysis. Only the input.h5 files are linked (same person files and node schema):

    python scripts/proj/make_prodv3.py --recalib VA MA WA FL
    python scripts/proj/state_pipeline.py submit run prodv4_FL prodv4_VA prodv4_WA prodv4_MA --tag prod-v4 \
        --peak-after 2026-11-01 --keep-n 60 --min-keep 45 --max-topups 3 --upscale-to 300 --no-us \
        --pass-rates ../flu-round1-2026-27/analysis/prod-60-v2/pass_rates.csv --no-calib-report \
        --partition bii-largemem --jobs 44 --extra-partition bii --extra-qos bii-half --extra-jobs 54

Links, not copies (prodv3): calib_prodv3_<ST>/ holds symlinks to the base's calibration entries (scores, runs: the
restart reads its source run's S0), proj_prodv3_<ST>/ copies of members.csv and restart/{selected,screened}.csv,
and the scratch restart input.h5 is a symlink to the base's (same node schema in v4).

GUARDED (--guarded -> prodv3g_<ST>, tag prod-v3g; 2026-11-06): prodv3_<ST>'s finished runs with prod-60-v2's guard
re-applied to their scenario-B seasons, no new simulation. A run is dropped when its B admissions peak (week
ending) before 2026-11-01 (proj.peak_ok) or by the size guard (proj.guard_weights: weight 1 up to 1.5x the
state's record season total / peak, cos^2 taper to 0 at 2x), applied as a Bernoulli keep with probability = weight
(seeded per state: the fixed-set equivalent of the production guard's PPS draw). proj_prodv3g_<ST>/restart/:
selected.csv (the kept), selected_unguarded.csv (prodv3's 60), guard_v3.csv (every run: peak week, total, peak,
record, weight, kept), and per-run symlinks to the kept members' parquet/json in every scenario; calib entries and
the input.h5 linked as for prodv3. Needs all of prodv3_<ST>'s runs (skipped with a message otherwise):

    python scripts/proj/make_prodv3.py --guarded VA MA WA FL
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py-modules"))
from flu_smh import experiment, proj, sim  # noqa: E402

#: prod-60-v2's experiment per state
BASE = {"FL": "seedx3_FL", "WA": "seedx2_WA", "CA": "seedx2_CA", "LA": "seedx2_LA", "MT": "seedx2_MT",
        "AL": "seedx2_AL", "TN": "seedx2_TN", "UT": "seedx2_UT"}
NAME = "prodv3_{}"
NAME_RECALIB = "prodv4_{}"
NAME_GUARDED = "prodv3g_{}"
PEAK_AFTER = "2026-11-01"
GUARD_SEED = 20261106
#: the base's round-2 design (members per calibration fit), by base-name prefix
PER_FIT = {"prod2025o": 16, "seedx2": 32, "seedx3": 64}
MODEL = "flu-r1-2026-27-v4.esl"
VACCINE = {"vax_escape_per_day": 1.0, "vax_infection_ve": 0.43, "vax_symp_ve": 0.46, "vax_hosp_ve": 0.0}
TAU_MULT = 1.16
HOSP_MULT = 1.08


def make(st: str, force: bool = False, recalib: bool = False) -> str:
    base = experiment.load(BASE.get(st, f"prod2025o_{st}"))
    name = (NAME_RECALIB if recalib else NAME).format(st)
    f = experiment.EXPERIMENTS / f"{name}.json"
    if f.exists() and not force:
        print(f"{name}: exists (--force to rewrite)")
        return name
    d = json.loads((experiment.EXPERIMENTS / f"{base.name}.json").read_text())
    d["name"] = name
    d["model"] = MODEL
    fixed = dict(d.get("fixed", {}))
    fixed |= VACCINE
    fixed["tau_mult"] = TAU_MULT
    fixed["hosp_prob_scale"] = float(fixed.get("hosp_prob_scale", 1.0)) * HOSP_MULT
    d["fixed"] = fixed
    d["keep_h5"] = False
    what = ("recalibrated from round 1 (state_pipeline), the base's priors x tau_mult" if recalib else
            "the same 60 selected members, not re-selected")
    d["notes"] = (f"{'prod-v4' if recalib else 'prod-v3'} test (2026-11-05): {base.name} under model v4 with the "
                  f"hub-matched vaccine {VACCINE}, tau x {TAU_MULT}, hosp_prob_scale x {HOSP_MULT}; {what}. "
                  "scripts/proj/make_prodv3.py. Base notes: " + d.get("notes", ""))
    d.setdefault("provenance", {})["prodv3_base"] = base.name
    if recalib:
        d["production"] = dict(d["production"], top=8, per_fit=PER_FIT[base.name.split("_")[0]])
    f.write_text(json.dumps(d, indent=1))
    exp = experiment.load(name)
    if recalib:
        for home, bhome in ((sim.input_path(exp), sim.input_path(experiment.load(f"prod2025o_{st}"))),
                            (proj.input_home(exp, "restart"), proj.input_home(base, "restart"))):
            if not bhome.exists():
                raise FileNotFoundError(bhome)
            home.parent.mkdir(parents=True, exist_ok=True)
            if not home.exists():
                home.symlink_to(bhome.resolve())
        print(f"{name}: from {base.name}; per_fit {d['production']['per_fit']}; inputs linked; "
              f"{exp.work_dir}")
        return name

    # calibration: symlinks to the base's entries
    cal, bcal = exp.out_dir, base.out_dir
    cal.mkdir(parents=True, exist_ok=True)
    for p in bcal.iterdir():
        q = cal / p.name
        if not q.exists() and not q.is_symlink():
            q.symlink_to(p)
    # projection: the selected members (and what the analyses read beside them)
    pd_, bpd = proj.out_dir(exp), proj.out_dir(base)
    (pd_ / "restart").mkdir(parents=True, exist_ok=True)
    for rel in ("members.csv", "restart/selected.csv", "restart/screened.csv"):
        if (bpd / rel).exists():
            shutil.copy2(bpd / rel, pd_ / rel)
    # the restart input.h5
    home, bhome = proj.input_home(exp, "restart"), proj.input_home(base, "restart")
    if not bhome.exists():
        raise FileNotFoundError(bhome)
    home.parent.mkdir(parents=True, exist_ok=True)
    if not home.exists():
        home.symlink_to(bhome.resolve())
    print(f"{name}: from {base.name}; {len(open(pd_ / 'restart/selected.csv').readlines()) - 1} members; "
          f"{exp.out_dir}, {pd_}, {home}")
    return name


def guard_table(src, sel: pd.DataFrame) -> pd.DataFrame:
    """prodv3 scenario-B season per selected member with the guard's verdict (see GUARDED)."""
    tt = (proj.member_totals(src, "restart", proj.scenario_id(src, "B")).set_index("member")
          .reindex(sel.member).reset_index())
    ok = proj.peak_ok(tt, PEAK_AFTER)
    w, rec = proj.guard_weights(src, tt)                     # size only; the timing cut is `ok`
    u = np.random.default_rng([GUARD_SEED, zlib.crc32(src.state.encode())]).random(len(tt))
    keep = ok & (u < w)
    cols = ["member", "proj_model_total", "proj_model_peak", "proj_model_peak_week"]
    g = tt[[c for c in cols if c in tt]].copy()
    g["record_total"], g["record_peak"] = rec.get("record_total", np.nan), rec.get("record_peak", np.nan)
    g["size_weight"], g["peak_ok"], g["draw"], g["kept"] = w, ok, u, keep
    g["why"] = np.where(keep, "", np.where(~ok, f"peak before {PEAK_AFTER}", "size guard"))
    return g


def make_guarded(st: str, force: bool = False) -> str | None:
    src = experiment.load(NAME.format(st))
    name = NAME_GUARDED.format(st)
    f = experiment.EXPERIMENTS / f"{name}.json"
    if f.exists() and not force:
        print(f"{name}: exists (--force to rewrite)")
        return name
    spd = proj.out_dir(src)
    sel = pd.read_csv(spd / "restart" / "selected.csv")
    sids = proj.scenario_ids(src)
    missing = [(s, m) for s in sids for m in sel.member if not proj.member_path(src, "restart", s, m).exists()]
    if missing:
        print(f"{name}: {src.name} has {len(missing)} of {len(sids) * len(sel)} runs missing; not made")
        return None
    g = guard_table(src, sel)
    d = json.loads((experiment.EXPERIMENTS / f"{src.name}.json").read_text())
    d["name"] = name
    d["notes"] = (f"prod-v3g (2026-11-06): {src.name}'s runs with prod-60-v2's guard re-applied to scenario B "
                  f"(peak on/after {PEAK_AFTER}, size taper 1.5-2x record as a Bernoulli keep); no new runs. "
                  f"{int(g.kept.sum())} of {len(g)} kept. scripts/proj/make_prodv3.py --guarded. "
                  + d.get("notes", ""))
    d.setdefault("provenance", {})["prodv3g_from"] = src.name
    f.write_text(json.dumps(d, indent=1))
    exp = experiment.load(name)
    # calibration entries (themselves links to the base's)
    cal = exp.out_dir
    cal.mkdir(parents=True, exist_ok=True)
    for p in src.out_dir.iterdir():
        q = cal / p.name
        if not q.exists() and not q.is_symlink():
            q.symlink_to(p.resolve())
    # projection: tables + links to the kept runs
    pd_ = proj.out_dir(exp)
    r = pd_ / "restart"
    r.mkdir(parents=True, exist_ok=True)
    for rel in ("members.csv", "restart/screened.csv"):
        if (spd / rel).exists():
            shutil.copy2(spd / rel, pd_ / rel)
    kept = sel[sel.member.isin(g.member[g.kept])]
    kept.to_csv(r / "selected.csv", index=False)
    sel.to_csv(r / "selected_unguarded.csv", index=False)
    g.to_csv(r / "guard_v3.csv", index=False)
    for s in sids:
        for m in sel.member:
            for ext in (".parquet", ".json"):
                p = proj.member_path(src, "restart", s, m).with_suffix(ext)
                q = proj.member_path(exp, "restart", s, m).with_suffix(ext)
                q.parent.mkdir(parents=True, exist_ok=True)
                if q.is_symlink() or q.exists():
                    q.unlink()
                if m in set(kept.member) and p.exists():
                    q.symlink_to(p.resolve())
    home, shome = proj.input_home(exp, "restart"), proj.input_home(src, "restart")
    home.parent.mkdir(parents=True, exist_ok=True)
    if not home.exists() and shome.exists():
        home.symlink_to(shome.resolve())
    why = g.why[g.why != ""].value_counts().to_dict()
    print(f"{name}: {len(kept)} of {len(sel)} kept ({', '.join(f'{v} {k}' for k, v in why.items()) or 'none dropped'})",
          flush=True)
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recalib", action="store_true", help="prodv4_<ST>: recalibrate (state_pipeline from round 1)")
    ap.add_argument("--guarded", action="store_true", help="prodv3g_<ST>: prodv3's runs, prod-60-v2's guard re-applied")
    a = ap.parse_args(argv)
    for st in a.states:
        if a.guarded:
            make_guarded(st.upper(), a.force)
        else:
            make(st.upper(), a.force, a.recalib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
