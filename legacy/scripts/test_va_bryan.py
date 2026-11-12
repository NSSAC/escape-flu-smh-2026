"""Run escape."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl
import epiweeks

from escape_abm.utils import OpenMPSimulator
from slurm_workflows import DsService, SlurmPilotExecutor, check_for_error

# SERVER_EXE = "apptainer run /project/bii_nssac/people/pb5gj/shared/ds-service/latest/ds-service.sif"
SERVER_EXE = "apptainer run /project/bii_nssac/people/pb5gj/shared/ds-service/2026-07-29-1/ds-service.sif"

SBATCH_ARGS = [
    "--account=bii_nssac",
    "--partition=bii --nodes=1",
    "--ntasks-per-node=1 --cpus-per-task=40 --mem=0",
    "--time=1:00:00",
]

ESCAPE_ENV = """
module load gcc/14.2.0
eval "$('/home/pb5gj/miniforge3/bin/conda' 'shell.bash' 'hook')"
conda activate hp13
"""

SIM_FILE = "/home/bl4zc/biocomplexity/SMH/escape-covid19-smh/models/covid19-r20.esl"
PERSON_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/person/{region}-person.parquet"
NETWORK_FILE_FMT = "/project/bii_nssac/EpiHiperSynPop/v2.4.0/from-alt-project/usa_{region}/{region}_contact_network.csv"
SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality.parquet"
DEATH_PROB_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/death-prob/{region}-death-prob.parquet"
WINTER_SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality-winter.parquet"
SUMMER_SEASONALITY_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/seasonality/{region}-seasonality-summer.parquet"
VACCINATION_FILE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/data/vaccination/{region}-{scenario}-vaccination.parquet"

SEASONALITY_FILE_FMT_BY_VARIANT = {
    "regular": SEASONALITY_FILE_FMT,
    "winter": WINTER_SEASONALITY_FILE_FMT,
    "summer": SUMMER_SEASONALITY_FILE_FMT,
}

WORK_DIR_FMT = "/scratch/bl4zc/covid19-round20/{region}/{scenario}/{seasonal_variant}"

SIM_START_DATE = pl.date(2025, 5, 25)
CALIB_START_DATE = pl.date(2025, 6, 8)
CALIB_END_DATE = pl.date(2026, 6, 6)
TARGET_ATTACK_RATE = 0.5

NUM_CALIB_TICKS = 378
NUM_PROJ_TICKS = 742

# Replicate simulations run per (region, scenario, seasonal variant).
NUM_REPLICATES = 15
REGIONS = ["VA"]
#SCENARIOS = ["A", "B", "C", "D", "E"]
SCENARIOS = ["A"]


# One of "regular", "winter", "summer".
SEASONAL_VARIANT = "regular"
# Variants to sweep over; add more entries to run several in one batch.
SEASONAL_VARIANTS = [SEASONAL_VARIANT]


TARGET_SCHEMA = {
    "location": pl.String(),
    "date": pl.Date(),
    "observation": pl.Float64(),
    "age_group": pl.String(),
    "target": pl.String(),
}
TARGET_DATA_FILE = "/project/bii_nssac/biocomplexity/SMH/covid19-scenario-modeling-hub-round20/target-data/time-series.csv"
LOCATION_DATA_FILE = "/project/bii_nssac/biocomplexity/SMH/covid19-scenario-modeling-hub-round20/auxiliary-data/data-locations/locations.csv"


def do_make_simulator(
    region: str, scenario: str, seasonal_variant: str
) -> OpenMPSimulator:
    if seasonal_variant not in SEASONALITY_FILE_FMT_BY_VARIANT:
        raise ValueError(
            f"Unknown seasonal variant: {seasonal_variant!r} "
            f"(expected one of {sorted(SEASONALITY_FILE_FMT_BY_VARIANT)})"
        )
    seasonality_file = SEASONALITY_FILE_FMT_BY_VARIANT[seasonal_variant].format(
        region=region
    )
    seasonality = pl.read_parquet(seasonality_file)
    seasonality = list(seasonality.iter_rows(named=True))

    death_prob_file = DEATH_PROB_FILE_FMT.format(region=region)
    death_prob = pl.read_parquet(death_prob_file)
    death_prob = list(death_prob.iter_rows(named=True))

    vaccination_file = VACCINATION_FILE_FMT.format(region=region, scenario=scenario)
    vax_schedule = pl.read_parquet(vaccination_file)
    vax_schedule = list(vax_schedule.iter_rows(named=True))

    work_dir = WORK_DIR_FMT.format(
        region=region, scenario=scenario, seasonal_variant=seasonal_variant
    )
    return OpenMPSimulator(
        SIM_FILE,
        work_dir,
        seasonality=seasonality,
        vax_schedule=vax_schedule,
        death_prob=death_prob,
    )


def do_prepare_scenario(region: str, scenario: str, seasonal_variant: str):
    simulator = do_make_simulator(region, scenario, seasonal_variant)

    simulator.prepare_build()
    simulator.build()

    person_file = PERSON_FILE_FMT.format(region=region)
    network_file = NETWORK_FILE_FMT.format(region=region.lower())
    work_dir = WORK_DIR_FMT.format(
        region=region, scenario=scenario, seasonal_variant=seasonal_variant
    )
    input_file = Path(work_dir) / "input.h5"

    simulator.prepare_input(person_file, network_file, input_file)


def do_run_simulation(
    region: str,
    scenario: str,
    seasonal_variant: str,
    num_ticks: int,
    configs: dict[str, Any],
) -> Path:
    simulator = do_make_simulator(region, scenario, seasonal_variant)
    work_dir = WORK_DIR_FMT.format(
        region=region, scenario=scenario, seasonal_variant=seasonal_variant
    )

    input_file = Path(work_dir) / "input.h5"
    output_file = Path(work_dir) / ("output-" + str(uuid4()) + ".h5")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    simulator.simulate(
        input_file, output_file, num_ticks=num_ticks, configs=configs, verbose=True
    )

    return output_file


def do_extract_pred_hosp(
    region: str, scenario: str, seasonal_variant: str, output_file: Path
) -> pl.DataFrame:
    simulator = do_make_simulator(region, scenario, seasonal_variant)

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
    region: str, scenario: str, seasonal_variant: str, output_file: Path
) -> float:
    pred_df = do_extract_pred_hosp(region, scenario, seasonal_variant, output_file)
    target_df = do_read_target_hosp(region)
    joint_df = target_df.join(pred_df, on="week_end_date").fill_null(0.0)

    y_true = joint_df["new_hosp_count"]
    y_pred = joint_df["new_hosp_count_pred"]
    rmse = np.sqrt(np.square(y_true - y_pred).mean()).item()
    return rmse


def do_calibration_run(region: str, configs: dict[str, Any]):
    output_file = do_run_simulation(region, "B", "regular", NUM_CALIB_TICKS, configs)
    hosp_rmse = do_compute_hosp_rmse(region, "B", "regular", output_file)

    result: dict[str, Any]
    result = {"region": region}
    result.update(configs)
    result["ouput_file"] = str(output_file)
    result["hosp_rmse"] = hosp_rmse
    return result



def test_sims():
    with DsService(server_exe=SERVER_EXE) as ds_service:
        ds_service.start()
        ds_service.wait_until_ready()

        executor = SlurmPilotExecutor(ds_service.get_address("ib0"))
        executor.define_worker(name="bii", sbatch_args=SBATCH_ARGS, setup_script=ESCAPE_ENV)

        try:
            executor.scale_workers("bii", 5)
            tasks = []
            for region in REGIONS:
                for scenario in SCENARIOS:
                    for seasonal_variant in SEASONAL_VARIANTS:
                        tasks.append(
                            executor.submit(
                                "bii",
                                do_prepare_scenario,
                                region,
                                scenario,
                                seasonal_variant,
                            )
                        )

            executor.wait(tasks)
            if check_for_error(tasks):
                return

            executor.scale_workers("bii", 25)
            tasks = []
            for region in REGIONS:
                for scenario in SCENARIOS:
                    for seasonal_variant in SEASONAL_VARIANTS:
                        for replicate in range(NUM_REPLICATES):
                            tasks.append(
                                executor.submit(
                                    "bii",
                                    do_run_simulation,
                                    region,
                                    scenario,
                                    seasonal_variant,
                                    NUM_PROJ_TICKS,
                                    {},
                                )
                            )

            executor.wait(tasks)
            if check_for_error(tasks):
                return
        finally:
            executor.scale_workers("bii", 0)


if __name__ == "__main__":
    test_sims()
