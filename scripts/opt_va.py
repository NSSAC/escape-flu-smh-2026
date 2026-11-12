"""Calibrate ESCAPE simulation against the SMH round-20 hospitalization target."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl
import epiweeks

from escape_abm.utils import OpenMPSimulator
from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor, check_for_error
from slurm_workflows.bayes_opt_botorch import BayesOptBotorch, FloatRange

SBATCH_ARGS = [
    "--account=bii_nssac --qos=bii-unlimited",
    "--partition=bii --nodes=1",
    "--ntasks-per-node=1 --cpus-per-task=40 --mem=0",
    "--time=6:00:00",
]

ESCAPE_ENV = """
module load gcc/14.2.0
eval "$('/home/pb5gj/miniforge3/bin/conda' 'shell.bash' 'hook')"
conda activate hp13
"""

# SIM_FILE = "/home/pb5gj/escape-covid19-smh/models/model.esl"
SIM_FILE = "/home/pb5gj/escape-covid19-smh/models/covid19-r20.esl"
PERSON_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/person/{region}-person.parquet"
NETWORK_FILE_FMT = "/project/bii_nssac/EpiHiperSynPop/v2.4.0/from-alt-project/usa_{region}/{region}_contact_network.csv"
SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality.parquet"
DEATH_PROB_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/death-prob/{region}-death-prob.parquet"
WINTER_SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality-winter.parquet"
VACCINATION_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/vaccination/{region}-{scenario}-vaccination.parquet"

WORK_DIR_FMT = "/scratch/pb5gj/covid19-round20/{region}/{scenario}/{seasonality_type}"

SIM_START_DATE = pl.date(2025, 5, 25)
CALIB_START_DATE = pl.date(2025, 6, 8)
CALIB_END_DATE = pl.date(2026, 6, 6)
TARGET_ATTACK_RATE = 0.5

NUM_CALIB_TICKS = 378
NUM_PROJ_TICKS = 742

TARGET_SCHEMA = {
    "location": pl.String(),
    "date": pl.Date(),
    "observation": pl.Float64(),
    "age_group": pl.String(),
    "target": pl.String(),
}
TARGET_DATA_FILE = "/project/bii_nssac/biocomplexity/SMH/covid19-scenario-modeling-hub-round20/target-data/time-series.csv"
LOCATION_DATA_FILE = "/project/bii_nssac/biocomplexity/SMH/covid19-scenario-modeling-hub-round20/auxiliary-data/data-locations/locations.csv"


def do_make_simulator(region: str, scenario: str, winter: bool) -> OpenMPSimulator:
    if winter:
        seasonality_file = WINTER_SEASONALITY_FILE_FMT.format(region=region)
        seasonality_type = "winter"
    else:
        seasonality_file = SEASONALITY_FILE_FMT.format(region=region)
        seasonality_type = "regular"
    seasonality = pl.read_parquet(seasonality_file)
    seasonality = list(seasonality.iter_rows(named=True))

    death_prob_file = DEATH_PROB_FILE_FMT.format(region=region)
    death_prob = pl.read_parquet(death_prob_file)
    death_prob = list(death_prob.iter_rows(named=True))

    vaccination_file = VACCINATION_FILE_FMT.format(region=region, scenario=scenario)
    vax_schedule = pl.read_parquet(vaccination_file)
    vax_schedule = list(vax_schedule.iter_rows(named=True))

    work_dir = WORK_DIR_FMT.format(
        region=region, scenario=scenario, seasonality_type=seasonality_type
    )
    return OpenMPSimulator(
        SIM_FILE,
        work_dir,
        seasonality=seasonality,
        vax_schedule=vax_schedule,
        death_prob=death_prob,
    )


def do_prepare_scenario(region: str, scenario: str, winter: bool):
    seasonality_type = "winter" if winter else "regular"
    simulator = do_make_simulator(region, scenario, winter)

    simulator.prepare_build()
    simulator.build()

    person_file = PERSON_FILE_FMT.format(region=region)
    network_file = NETWORK_FILE_FMT.format(region=region.lower())
    work_dir = WORK_DIR_FMT.format(
        region=region, scenario=scenario, seasonality_type=seasonality_type
    )
    input_file = Path(work_dir) / "input.h5"

    simulator.prepare_input(person_file, network_file, input_file)


def do_run_simulation(
    region: str,
    scenario: str,
    winter: bool,
    num_ticks: int,
    configs: dict[str, Any],
) -> Path:
    seasonality_type = "winter" if winter else "regular"
    simulator = do_make_simulator(region, scenario, winter)
    work_dir = WORK_DIR_FMT.format(
        region=region, scenario=scenario, seasonality_type=seasonality_type
    )

    input_file = Path(work_dir) / "input.h5"
    output_file = Path(work_dir) / ("output-" + str(uuid4()) + ".h5")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    simulator.simulate(
        input_file, output_file, num_ticks=num_ticks, configs=configs, verbose=True
    )

    return output_file


def do_extract_pred_hosp(
    region: str, scenario: str, winter: bool, output_file: Path
) -> pl.DataFrame:
    simulator = do_make_simulator(region, scenario, winter)

    statistics = (
        simulator.extract_statistics(output_file)
        .with_columns((SIM_START_DATE + pl.duration(days="tick")).alias("date"))
        .filter(pl.col("date") >= CALIB_START_DATE, pl.col("date") <= CALIB_END_DATE)
        .with_columns(
            pl.col("date")
            .map_elements(
                lambda d: epiweeks.Week.fromdate(d).enddate(), return_dtype=pl.Date
            )
            .alias("week_end_date")
        )
    )

    statistics_weekly = (
        statistics.group_by("week_end_date")
        .agg(pl.sum("new_hosp_count").alias("new_hosp_count_pred"))
        .sort("week_end_date")
    )

    return statistics_weekly


def do_read_target_hosp(region: str) -> pl.DataFrame:
    target_df = pl.read_csv(TARGET_DATA_FILE, schema_overrides=TARGET_SCHEMA)
    location_df = pl.read_csv(LOCATION_DATA_FILE)

    new_target_df = (
        target_df.join(location_df, on="location")
        .filter(
            pl.col("abbreviation") == region,
            pl.col("age_group") == "0-130",
            pl.col("target") == "inc hosp",
            pl.col("date") >= CALIB_START_DATE,
            pl.col("date") <= CALIB_END_DATE,
        )
        .sort("date")
        .select("date", "observation")
        .rename({"date": "week_end_date", "observation": "new_hosp_count"})
    )
    return new_target_df


def do_compute_hosp_rmse(
    region: str, scenario: str, winter: bool, output_file: Path
) -> float:
    pred_df = do_extract_pred_hosp(region, scenario, winter, output_file)
    target_df = do_read_target_hosp(region)
    joint_df = target_df.join(pred_df, on="week_end_date").fill_null(0.0)

    y_true = joint_df["new_hosp_count"]
    y_pred = joint_df["new_hosp_count_pred"]
    rmse = np.sqrt(np.square(y_true - y_pred).mean()).item()
    return rmse


def do_calibration_run(region: str, configs: dict[str, Any]):
    output_file = do_run_simulation(region, "B", False, NUM_CALIB_TICKS, configs)
    hosp_rmse = do_compute_hosp_rmse(region, "B", False, output_file)

    result: dict[str, Any]
    result = {"region": region}
    result.update(configs)
    result["output_file"] = str(output_file)
    result["hosp_rmse"] = hosp_rmse
    return result


JOB_NAME = "VA_OPT"

REGION = "WA"
CALIB_SCENARIO = "B"
CALIB_WINTER = False

SEED = 20260803

SEARCH_SPACE = {
    "baseline_tau": FloatRange(0.007, 0.021),
    "tau_seasonality_scale": FloatRange(0.001, 0.01),
}

EXPLORATION_POINTS = 64
SEARCH_PARALLELISM = 10

MIN_SEARCH_ITERATIONS = 5
MAX_SEARCH_ITERATIONS = 10
PATIENCE = 3
MIN_IMPROVEMENT = 0.05


def objective(
    baseline_tau: float,
    tau_seasonality_scale: float,
    *,
    region: str,
) -> dict[str, Any]:
    """One calibration run. This is what executes on a compute node.

    Its argument names have to match the keys of `SEARCH_SPACE`;
    `region` is not searched over,
    so it arrives as one of the optimizer's extra objective kwargs.

    `do_calibration_run` already returns the mapping the optimizer wants:
    the value to minimize under `hosp_rmse`
    --- named to the optimizer through `objective_key` below ---
    alongside the configs it ran at and the output file it wrote,
    which are recorded but not modelled.
    """
    return do_calibration_run(
        region,
        configs=dict(
            baseline_tau=baseline_tau,
            tau_seasonality_scale=tau_seasonality_scale,
        ),
    )


def main():
    with DsServiceServer(interface="ib0") as ds_service:
        ds_service.wait_until_ready()

        # Leaving this block cancels every pilot job,
        # including if the body raises.
        with SlurmPilotExecutor(ds_service.address) as executor:
            executor.define_worker(
                name="bii", sbatch_args=SBATCH_ARGS, setup_script=ESCAPE_ENV
            )

            executor.scale_workers("bii", 1)
            tasks = [
                executor.submit(
                    "bii", do_prepare_scenario, REGION, CALIB_SCENARIO, CALIB_WINTER
                )
            ]
            executor.wait(tasks, desc="prepare")
            if check_for_error(tasks):
                raise RuntimeError("preparing the calibration scenario failed")

            opt = BayesOptBotorch(
                JOB_NAME,
                SEARCH_SPACE,
                objective,
                executor,
                "bii",
                "bii",
                EXPLORATION_POINTS,
                SEARCH_PARALLELISM,
                SEED,
                min_search_iterations=MIN_SEARCH_ITERATIONS,
                max_search_iterations=MAX_SEARCH_ITERATIONS,
                patience=PATIENCE,
                min_improvement=MIN_IMPROVEMENT,
                objective_key="hosp_rmse",
                # Everything past here is forwarded to the objective
                # unchanged: the constants it needs
                # but the search should not vary.
                region=REGION,
            )

            executor.scale_workers("bii", EXPLORATION_POINTS)
            print(
                f"\n=== exploration: {opt.num_exploration_points} points, one batch ==="
            )
            opt.run_exploration_jobs()

            executor.scale_workers("bii", SEARCH_PARALLELISM)
            print(
                f"\n=== search: up to {MAX_SEARCH_ITERATIONS} rounds"
                f" of {SEARCH_PARALLELISM} points ==="
            )
            opt.run_search_jobs()

    params, value = opt.best_point()
    print(f"\nbest hosp_rmse = {value:.6g}")
    print(
        f"  found at baseline_tau = {params['baseline_tau']:.6g}, "
        f"tau_seasonality_scale = {params['tau_seasonality_scale']:.6g}"
    )
    print(f"  full result: {opt.best_output()}")


if __name__ == "__main__":
    main()
