"""Explore the parameter space."""

import shutil
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
    "--time=2-00:00:00",
]

NUM_NODES = 90

SETUP_SCRIPT = """
module load gcc/14.2.0
eval "$(/home/pb5gj/miniforge3/bin/conda shell.bash hook)"
conda activate hp13
"""

SIM_FILE = "/home/pb5gj/escape-covid19-smh/models/covid19-r20.esl"
PERSON_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/person/{region}-person.parquet"
NETWORK_FILE_FMT = "/project/bii_nssac/EpiHiperSynPop/v2.4.0/from-alt-project/usa_{region}/{region}_contact_network.csv"
SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality.parquet"
DEATH_PROB_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/death-prob/{region}-death-prob.parquet"
WINTER_SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality-winter.parquet"
VACCINATION_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/vaccination/{region}-{scenario}-vaccination.parquet"

OUTPUT_DIR_FMT = (
    "/scratch/pb5gj/covid19-round20-calib-4/{region}/{scenario}/{seasonality_type}"
)

SIM_START_DATE = pl.date(2025, 5, 25)
CALIB_START_DATE = pl.date(2025, 6, 8)
CALIB_END_DATE = pl.date(2026, 6, 6)

NUM_CALIB_TICKS = 378

TARGET_SCHEMA = {
    "location": pl.String(),
    "date": pl.Date(),
    "observation": pl.Float64(),
    "age_group": pl.String(),
    "target": pl.String(),
}
TARGET_DATA_FILE = "/project/bii_nssac/biocomplexity/SMH/covid19-scenario-modeling-hub-round20/target-data/time-series.csv"
LOCATION_DATA_FILE = "/project/bii_nssac/biocomplexity/SMH/covid19-scenario-modeling-hub-round20/auxiliary-data/data-locations/locations.csv"

JOB_NAME = "explore-param-space"
RESULTS_FILE = "exploration-results.parquet"

# REGIONS = ["KY"]
# REGIONS = """
# AL AR AZ CA CO CT
# DC DE FL GA HI IA
# ID IL IN KS KY LA
# MA MD ME MI MN MO
# MS MT NC ND NE NH
# NJ NM NV NY OH OK
# OR PA RI SC SD TN
# TX UT VA VT WA WI
# WV WY
# """.split()
REGIONS = """
CA DC DE FL GA
HI LA NV SC TX
WA WY
""".split()

CALIB_SCENARIO = "B"
CALIB_WINTER = False

SEED = 20260803

SEARCH_SPACE = {
    "winter_seasonality_scale": FloatRange(1e-09, 2.2219),
    "tau_seasonality_scale": FloatRange(0.00387302, 0.0202083),
    "baseline_tau": FloatRange(0.00776265, 0.0193971),
    "hosp_prob_scale": FloatRange(0.652281, 1.86639),
    "hosp_prob_65p_scale": FloatRange(0.718008, 2.38711),
    "infection_waning_per_day": FloatRange(0.989985, 0.996975),
}

EXPLORATION_POINTS = 1024
SEARCH_PARALLELISM = 10

MIN_SEARCH_ITERATIONS = 5
MAX_SEARCH_ITERATIONS = 10
PATIENCE = 3
MIN_IMPROVEMENT = 0.03


def do_get_work_dir(
    region: str,
    scenario: str,
    winter: bool,
    delete_if_exists: bool = False,
    create_if_not_exists: bool = False,
) -> Path:
    seasonality_type = "winter" if winter else "regular"

    work_dir = Path(
        OUTPUT_DIR_FMT.format(
            region=region, scenario=scenario, seasonality_type=seasonality_type
        )
    )

    if delete_if_exists and work_dir.exists():
        shutil.rmtree(work_dir)

    if create_if_not_exists and not work_dir.exists():
        work_dir.mkdir(parents=True)

    return work_dir


def do_make_simulator(region: str, scenario: str, winter: bool) -> OpenMPSimulator:
    if winter:
        seasonality_file = WINTER_SEASONALITY_FILE_FMT.format(region=region)
    else:
        seasonality_file = SEASONALITY_FILE_FMT.format(region=region)
    seasonality = pl.read_parquet(seasonality_file)
    seasonality = list(seasonality.iter_rows(named=True))

    death_prob_file = DEATH_PROB_FILE_FMT.format(region=region)
    death_prob = pl.read_parquet(death_prob_file)
    death_prob = list(death_prob.iter_rows(named=True))

    vaccination_file = VACCINATION_FILE_FMT.format(region=region, scenario=scenario)
    vax_schedule = pl.read_parquet(vaccination_file)
    vax_schedule = list(vax_schedule.iter_rows(named=True))

    work_dir = do_get_work_dir(region, scenario, winter)

    return OpenMPSimulator(
        SIM_FILE,
        work_dir,
        seasonality=seasonality,
        vax_schedule=vax_schedule,
        death_prob=death_prob,
    )


def do_prepare_scenario(region: str, scenario: str, winter: bool) -> None:
    work_dir = do_get_work_dir(
        region, scenario, winter, delete_if_exists=True, create_if_not_exists=True
    )

    # Build simulator
    simulator = do_make_simulator(region, scenario, winter)
    simulator.prepare_build()
    simulator.build()

    # Create input file
    person_file = PERSON_FILE_FMT.format(region=region)
    network_file = NETWORK_FILE_FMT.format(region=region.lower())
    input_file = work_dir / "input.h5"
    simulator.prepare_input(person_file, network_file, input_file)


def do_run_simulation(
    region: str,
    scenario: str,
    winter: bool,
    num_ticks: int,
    configs: dict[str, Any],
) -> Path:
    simulator = do_make_simulator(region, scenario, winter)
    work_dir = do_get_work_dir(region, scenario, winter)

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
        # Adding a duration to a date widens it to a datetime;
        # narrow it back, so `date` is one
        # and the target data joins against it as one.
        .with_columns(
            (SIM_START_DATE + pl.duration(days="tick")).dt.date().alias("date")
        )
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
    joint_df = target_df.join(pred_df, on="week_end_date")

    y_true = joint_df["new_hosp_count"]
    y_pred = joint_df["new_hosp_count_pred"]
    rmse = np.sqrt(np.square(y_true - y_pred).mean()).item()
    return rmse


def do_calibration_run(region: str, configs: dict[str, Any]) -> dict[str, Any]:
    """One simulation and its rmse against the target data.

    The scenario is whichever one `main` prepared:
    running anything else here would score against a work dir
    nothing built, or --- worse, since a stale work dir usually survives
    on scratch --- against one an earlier calibration left behind.
    """
    output_file = do_run_simulation(
        region, CALIB_SCENARIO, CALIB_WINTER, NUM_CALIB_TICKS, configs
    )
    hosp_rmse = do_compute_hosp_rmse(region, CALIB_SCENARIO, CALIB_WINTER, output_file)

    result: dict[str, Any]
    result = {"region": region, "scenario": CALIB_SCENARIO, "winter": CALIB_WINTER}
    result.update(configs)
    result["output_file"] = str(output_file)
    result["hosp_rmse"] = hosp_rmse
    return result


def do_save_exploration_results(
    region: str, scenario: str, winter: bool, outputs: list[dict[str, Any]]
) -> Path:
    """Write every evaluated point to the work dir.

    The optimizer holds the objective's results in memory only,
    so without this the mapping from parameters to rmse
    is gone when the driver exits:
    the simulation outputs left on scratch record the configs they ran at
    but not the score they earned.

    Sorted best first,
    so the head of the file is the part worth reading.
    """
    work_dir = do_get_work_dir(region, scenario, winter)
    results_file = work_dir / RESULTS_FILE

    results_df = pl.DataFrame(outputs).sort("hosp_rmse")
    results_df.write_parquet(results_file)

    return results_file


def objective(
    baseline_tau: float,
    tau_seasonality_scale: float,
    hosp_prob_scale: float,
    hosp_prob_65p_scale: float,
    infection_waning_per_day: float,
    winter_seasonality_scale: float,
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
            hosp_prob_scale=hosp_prob_scale,
            hosp_prob_65p_scale=hosp_prob_65p_scale,
            infection_waning_per_day=infection_waning_per_day,
            winter_seasonality_scale=winter_seasonality_scale,
        ),
    )


def main() -> None:
    with DsServiceServer(interface="ib0") as ds_service:
        ds_service.wait_until_ready()

        with SlurmPilotExecutor(ds_service.address) as executor:
            executor.define_worker(
                name="bii", sbatch_args=SBATCH_ARGS, setup_script=SETUP_SCRIPT
            )
            executor.scale_workers("bii", NUM_NODES)

            tasks = []
            for region in REGIONS:
                tasks.append(
                    executor.submit(
                        "bii", do_prepare_scenario, region, CALIB_SCENARIO, CALIB_WINTER
                    )
                )

            executor.wait(tasks, desc="prepare")
            if check_for_error(tasks):
                raise RuntimeError("preparing the calibration scenario failed")

            for region in REGIONS:
                print(f"exploring: {region}")

                opt = BayesOptBotorch(
                    # Per region: the name prefixes every progress line
                    # the optimizer prints,
                    # which a multi-region run needs to tell them apart.
                    f"{JOB_NAME}-{region}",
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
                    region=region,
                )

                opt.run_exploration_jobs()

                results_file = do_save_exploration_results(
                    region, CALIB_SCENARIO, CALIB_WINTER, opt.outputs
                )
                print(f"saved {len(opt.outputs)} results: {results_file}")


if __name__ == "__main__":
    main()
