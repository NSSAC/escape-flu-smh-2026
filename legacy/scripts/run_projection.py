"""Run projection."""

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from escape_abm.utils import OpenMPSimulator
from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor, check_for_error

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

REPLICATE_FMT = "/project/bii_nssac/biocomplexity/SMH/covid19-round20/calib_particle_filtering-4/{region}/replicates.csv"

OUTPUT_DIR_FMT = (
    "/scratch/pb5gj/covid19-round20-proj-from-calib_particle_filtering-4/{region}/{scenario}/{seasonality_type}"
)

NUM_PROJ_TICKS = 742

# REGIONS = ["AZ"]
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


PROJ_SCENARIOS = ["A", "B", "C", "D", "E"]
PROJ_WINTER = False


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
    replicate_id: int,
    num_ticks: int,
    configs: dict[str, Any],
) -> Path:
    simulator = do_make_simulator(region, scenario, winter)
    work_dir = do_get_work_dir(region, scenario, winter)

    input_file = Path(work_dir) / "input.h5"
    output_file = Path(work_dir) / (f"output-{replicate_id}-" + str(uuid4()) + ".h5")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    simulator.simulate(
        input_file, output_file, num_ticks=num_ticks, configs=configs, verbose=True
    )

    return output_file


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
                for scenario in PROJ_SCENARIOS:
                    tasks.append(
                        executor.submit(
                            "bii", do_prepare_scenario, region, scenario, PROJ_WINTER
                        )
                    )

            executor.wait(tasks, desc="prepare")
            if check_for_error(tasks):
                raise RuntimeError("preparing the calibration scenario failed")

            tasks = []
            for region in REGIONS:
                print(f"exploring: {region}")

                for scenario in PROJ_SCENARIOS:
                    replicate_file = REPLICATE_FMT.format(region=region)
                    replicate_df = pl.read_csv(replicate_file)

                    for configs in replicate_df.iter_rows(named=True):
                        configs = dict(configs)
                        replicate_id = configs.pop("replicate_id")

                        tasks.append(
                            executor.submit(
                                "bii",
                                do_run_simulation,
                                region,
                                scenario,
                                PROJ_WINTER,
                                replicate_id,
                                NUM_PROJ_TICKS,
                                configs,
                            )
                        )

            executor.wait(tasks, desc="prepare")
            if check_for_error(tasks):
                raise RuntimeError("preparing the calibration scenario failed")


if __name__ == "__main__":
    main()
