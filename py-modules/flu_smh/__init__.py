"""Drivers for models/flu-r1-2026-27.esl: experiments, vaccination schedules, runs, calibration.

    experiment.py   an experiment = one state, one time window, one person file, the in-sim campaigns,
                    the parameter priors (experiments/<name>.json)
    schedule.py     hub coverage curves x the person file's ages -> the ESL's vax_schedule rows
    sim.py          render/build the model, prepare input.h5, run one parameter set, stream the HDF5
                    output into a small weekly table (admissions, infections, deaths by subtype x age)
    calib.py        Sobol particles over the priors, NHSN likelihood scoring, ranking

Command line: scripts/calib/flu_calib.py (prepare | sample | run | score | status).
"""
