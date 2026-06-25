# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Taxi Trip Data Forensics

Complete the `solve()` function. Read problem.md for the full specification.
Your submission is auto-graded against strict invariants — every impossible
row, missing null, or wrong column will fail a check.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Clean the raw taxi-trip log into a 16-column analysis-ready table.

    See problem.md for the exact column list, parsing rules, plausibility
    filters, payment-normalisation mapping, imputation, dedup rule, and the
    four derived columns. Return the cleaned frame sorted by pickup_datetime.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "sg_taxi_trips.parquet")

    columns = [
        "trip_id",
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_zone",
        "dropoff_zone",
        "distance_km",
        "fare_sgd",
        "tip_sgd",
        "payment_type",
        "passengers",
        "pickup_latitude",
        "pickup_longitude",
        "trip_duration_min",
        "implied_speed_kmh",
        "fare_per_km",
        "is_airport",
    ]

    df = (
        df.with_columns(
            pl.col("pickup_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col("dropoff_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col("tip_sgd").fill_null(0.0),
            pl.col("pickup_zone").fill_null("Unknown"),
            pl.col("dropoff_zone").fill_null("Unknown"),
            pl.when(pl.col("payment_type").str.to_lowercase().str.contains("grab"))
            .then(pl.lit("Grab"))
            .when(pl.col("payment_type").str.to_lowercase().str.contains("nets"))
            .then(pl.lit("NETS"))
            .when(pl.col("payment_type").str.to_lowercase().str.contains("cash"))
            .then(pl.lit("Cash"))
            .when(
                pl.col("payment_type")
                .str.to_lowercase()
                .str.contains("card|visa|mastercard|credit")
            )
            .then(pl.lit("Card"))
            .otherwise(pl.col("payment_type"))
            .alias("payment_type"),
        )
        .with_columns(
            (
                (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()
                / 60
            ).alias("trip_duration_min")
        )
        .with_columns(
            (pl.col("distance_km") / (pl.col("trip_duration_min") / 60)).alias(
                "implied_speed_kmh"
            )
        )
        .filter(
            (pl.col("fare_sgd") > 0)
            & (pl.col("distance_km") > 0)
            & (pl.col("distance_km") <= 100)
            & (pl.col("passengers") >= 1)
            & (pl.col("trip_duration_min") > 0)
            & (pl.col("trip_duration_min") <= 180)
            & (pl.col("implied_speed_kmh") >= 2)
            & (pl.col("implied_speed_kmh") <= 120)
        )
        .sort(["trip_id", "fare_sgd", "dropoff_datetime"], descending=[False, True, True])
        .unique(subset=["trip_id"], keep="first", maintain_order=True)
        .with_columns(
            (pl.col("fare_sgd") / pl.col("distance_km")).alias("fare_per_km"),
            (
                (pl.col("pickup_zone") == "Changi Airport")
                | (pl.col("dropoff_zone") == "Changi Airport")
            ).alias("is_airport"),
        )
        .select(columns)
        .sort("pickup_datetime")
    )

    return df


if __name__ == "__main__":
    print(solve().head())
