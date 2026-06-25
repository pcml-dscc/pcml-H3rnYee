# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 4: Profile, Clean & Integrate with DataExplorer

Complete the `solve()` function. Read problem.md for the full specification.
This task uses the kailash-ml DataExplorer engine to PROVE your cleaning
improved data quality.

    python grader.py starter.py
"""
from __future__ import annotations

import asyncio

import polars as pl

from kailash_ml import DataExplorer
from shared import MLFPDataLoader


def solve() -> dict:
    """Return {"cleaned": pl.DataFrame, "raw_alert_count": int, "clean_alert_count": int}.

    See problem.md for the exact 8-column cleaned schema, the THREE period
    formats you must parse, the comma-stripped integer cast, the median
    imputation rule, and the DataExplorer alert-count requirement.
    """
    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")

    quarterly = raw.filter(pl.col("period_type") == "quarterly")

    cleaned = (
        quarterly.with_columns(
            pl.coalesce(
                pl.col("period").str.extract(r"^Q([1-4])\s+\d{4}$", 1),
                pl.col("period").str.extract(r"^\d{4}-Q([1-4])$", 1),
                pl.col("period").str.extract(r"^\d{4}-([1-4])$", 1),
            )
            .cast(pl.Int64)
            .alias("period_quarter"),
            pl.coalesce(
                pl.col("period").str.extract(r"^Q[1-4]\s+(\d{4})$", 1),
                pl.col("period").str.extract(r"^(\d{4})-Q[1-4]$", 1),
                pl.col("period").str.extract(r"^(\d{4})-[1-4]$", 1),
            )
            .cast(pl.Int64)
            .alias("period_year"),
            pl.col("tourist_arrivals")
            .cast(pl.Utf8)
            .str.replace_all(",", "")
            .cast(pl.Int64),
        )
        .with_columns(
            pl.col("inflation_rate").fill_null(pl.col("inflation_rate").median()),
            pl.col("trade_balance_sgd_bn").fill_null(
                pl.col("trade_balance_sgd_bn").median()
            ),
        )
        .select(
            [
                "period_year",
                "period_quarter",
                "gdp_growth_pct",
                "unemployment_rate",
                "inflation_rate",
                "trade_balance_sgd_bn",
                "property_price_index",
                "tourist_arrivals",
            ]
        )
        .sort(["period_year", "period_quarter"])
    )

    async def _profile_counts() -> tuple[int, int]:
        explorer = DataExplorer()
        raw_profile = await explorer.profile(quarterly)
        clean_profile = await explorer.profile(cleaned)
        return len(raw_profile.alerts), len(clean_profile.alerts)

    raw_alert_count, clean_alert_count = asyncio.run(_profile_counts())

    return {
        "cleaned": cleaned,
        "raw_alert_count": raw_alert_count,
        "clean_alert_count": clean_alert_count,
    }


if __name__ == "__main__":
    print(solve())
