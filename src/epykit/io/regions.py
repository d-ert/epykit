"""
epykit.io.regions
==================
Helpers for working with BED-style regions (0-based, half-open).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl

BED_COLUMNS = ["chr", "start", "end"]


def read_bed_regions(path: str | Path) -> pd.DataFrame:
    """Read a BED file and return a DataFrame with chr/start/end.

    Parameters
    ----------
    path:
        BED file path. The file must be 0-based, half-open. Extra columns
        are ignored. Comment lines (starting with '#') are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        names=BED_COLUMNS,
    )
    return df


def merge_bed_intervals(regions: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or adjacent intervals per chromosome.

    Parameters
    ----------
    regions:
        DataFrame with columns chr/start/end (0-based, half-open).
    """
    if regions.empty:
        return regions.copy()

    regions = regions.sort_values(["chr", "start", "end"], kind="mergesort")
    merged_rows: list[tuple[str, int, int]] = []

    for chrom, group in regions.groupby("chr", sort=False):
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()

        cur_start = int(starts[0])
        cur_end = int(ends[0])

        for start, end in zip(starts[1:], ends[1:]):
            start = int(start)
            end = int(end)
            if start <= cur_end:  # overlap or adjacency (half-open)
                cur_end = max(cur_end, end)
            else:
                merged_rows.append((chrom, cur_start, cur_end))
                cur_start, cur_end = start, end

        merged_rows.append((chrom, cur_start, cur_end))

    return pd.DataFrame(merged_rows, columns=BED_COLUMNS)


def bed_to_polars(regions: pd.DataFrame) -> pl.DataFrame:
    """Convert a BED regions DataFrame to a Polars DataFrame."""
    if regions.empty:
        return pl.DataFrame({"chr": [], "start": [], "end": []})
    return pl.DataFrame(
        {
            "chr": regions["chr"].astype(str).to_list(),
            "start": regions["start"].astype("int32").to_list(),
            "end": regions["end"].astype("int32").to_list(),
        }
    )


def filter_lazyframe_to_regions(
    lf: pl.LazyFrame,
    regions: pl.DataFrame,
) -> pl.LazyFrame:
    """Filter a LazyFrame to BED regions using join_asof containment.

    The regions must be non-overlapping per chromosome (merge first).
    """
    if regions.is_empty():
        return lf.filter(pl.lit(False))

    regions = regions.sort(["chr", "start"]).rename(
        {"start": "region_start", "end": "region_end"}
    )

    return (
        lf.sort(["chr", "start"])
        .join_asof(
            regions.lazy(),
            left_on="start",
            right_on="region_start",
            by="chr",
            strategy="backward",
        )
        .filter(pl.col("start") < pl.col("region_end"))
        .drop(["region_start", "region_end"])
    )


def load_and_merge_regions(path: str | Path) -> pl.DataFrame:
    """Read BED and return merged regions as a Polars DataFrame."""
    regions = read_bed_regions(path)
    merged = merge_bed_intervals(regions)
    return bed_to_polars(merged)


def ensure_regions_dataframe(regions: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Ensure regions are a Polars DataFrame with chr/start/end int32."""
    if isinstance(regions, pd.DataFrame):
        return bed_to_polars(regions)
    if regions.is_empty():
        return regions
    return regions.with_columns(
        pl.col("chr").cast(pl.Utf8),
        pl.col("start").cast(pl.Int32),
        pl.col("end").cast(pl.Int32),
    )


def validate_bed_regions(regions: Iterable[tuple[str, int, int]]) -> None:
    """Validate BED regions for basic correctness."""
    for chrom, start, end in regions:
        if start < 0 or end < 0:
            raise ValueError("BED coordinates must be non-negative")
        if end < start:
            raise ValueError("BED end must be >= start")