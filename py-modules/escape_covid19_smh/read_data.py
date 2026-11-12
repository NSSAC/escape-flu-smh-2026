"""Read data frames."""

from pathlib import Path

import polars as pl


def read_location_data(smh_repo_path: Path) -> pl.DataFrame:
    schema = {
        "location": pl.String,
        "population": pl.Int64,
        "age_group": pl.Categorical,
        "abbreviation": pl.String,
        "location_name": pl.String,
    }

    return pl.read_csv(
        smh_repo_path / "auxiliary-data/data-locations/locations_2022.csv",
        schema_overrides=schema,
        null_values=["NA"],
    ).drop_nulls()


def read_target_data(smh_repo_path: Path) -> pl.DataFrame:
    schema = {
        "location": pl.String,
        "date": pl.Date,
        "observation": pl.Int64,
        "age_group": pl.String,
        "target": pl.Categorical,
    }
    return pl.read_csv(
        smh_repo_path / "target-data/time-series.csv", schema_overrides=schema
    ).drop_nulls()


def add_location_abbrev(
    target_data: pl.DataFrame, location_data: pl.DataFrame
) -> pl.DataFrame:
    ldf = location_data.filter(pl.col("age_group") == "0-130").select("location", "abbreviation")
    target_data = target_data.join(ldf, on="location")
    return target_data
