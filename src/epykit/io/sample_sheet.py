"""
epykit.io.sample_sheet
========================
Multi-sample loader: reads a CSV sample sheet, parallelises per-sample
file ingestion using Polars native threading, outer-joins on genomic loci,
and constructs a cohort-level AnnData object.

Sample sheet format
-------------------
A CSV with at minimum these columns::

    sample_id, path, group

Optional columns are passed through to ``adata.obs``:

    batch, age, sex, treatment, ...

Any column whose name is not ``sample_id`` or ``path`` is treated as
sample metadata and stored in ``adata.obs``.

Supported formats (auto-detected from file extension)
------------------------------------------------------
    .cov / .bismark.cov / .bismark.cov.gz   -> Bismark coverage
    .CX_report.txt / .CX_report.txt.gz      -> Bismark CX_report
    .bedgraph / .bedgraph.gz                -> bedGraph
    other                                   -> generic TSV

Usage
-----
>>> from epykit.io import read_samples
>>> adata = read_samples("cohort_sample_sheet.csv", min_coverage=5)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl

from epykit.io.anndata_builder import build_anndata
from epykit.io.bismark import read_bismark_coverage, read_bismark_cx_report
from epykit.io.generic import read_bedgraph, read_generic_methylation

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# AnnData is optional for type hints — import lazily
try:
    import anndata as ad
    AnnData = ad.AnnData
except ImportError:  # pragma: no cover
    AnnData = None  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    """Return a format key based on the file extension."""
    name = path.name.lower()
    if "cx_report" in name:
        return "cx_report"
    suffixes = "".join(path.suffixes).lower()
    if ".bismark.cov" in suffixes or suffixes.endswith(".cov"):
        return "bismark_cov"
    if ".bedgraph" in suffixes or ".bdg" in suffixes:
        return "bedgraph"
    return "generic"


def _load_single_sample(
    sample_id: str,
    path: Path,
    fmt: str,
    min_coverage: int,
    max_coverage: int | None,
    context: str | None,
    reader_kwargs: dict,
) -> tuple[str, pl.DataFrame]:
    """Load a single sample file and return (sample_id, DataFrame)."""
    logger.debug("Loading sample '%s' from %s (format=%s)", sample_id, path, fmt)

    if fmt == "bismark_cov":
        df = read_bismark_coverage(
            path,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            **reader_kwargs,
        )
        # Add context column if not present (Bismark .cov = CpG only)
        if "context" not in df.columns:
            df = df.with_columns(pl.lit("CpG").alias("context"))
        if "strand" not in df.columns:
            df = df.with_columns(pl.lit("*").alias("strand"))

    elif fmt == "cx_report":
        df = read_bismark_cx_report(
            path,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            context=context,
            **reader_kwargs,
        )

    elif fmt == "bedgraph":
        df = read_bedgraph(path, **reader_kwargs)
        df = df.with_columns(
            pl.lit("*").alias("strand"),
            pl.lit("CpG").alias("context"),
        )

    else:
        df = read_generic_methylation(
            path,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            **reader_kwargs,
        )

    # Ensure locus key columns exist
    for col in ("chr", "start", "end", "strand"):
        if col not in df.columns:
            raise ValueError(
                f"Sample '{sample_id}': required column '{col}' not found "
                f"after reading {path}."
            )

    return sample_id, df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_samples(
    sample_sheet: PathLike,
    *,
    min_coverage: int = 1,
    max_coverage: int | None = None,
    context: str | None = None,
    fmt: str | None = None,
    join_type: str = "outer",
    n_workers: int | None = None,
    reader_kwargs: dict | None = None,
    engine: str = "polars",
    output: str = "memory",
    out_path: str | Path | None = None,
    duckdb_memory_limit: str = "2GB",
    duckdb_threads: int | None = None,
) -> "AnnData":
    """Load a cohort of methylation samples and build an AnnData object.

    Reads a CSV sample sheet, loads each sample file in parallel using
    Polars native multithreading, performs a genomic outer join on the
    ``(chr, start, end, strand)`` locus key, and constructs a cohort-level
    AnnData object ready for QC and differential analysis.

    Parameters
    ----------
    sample_sheet:
        Path to the sample sheet CSV.  Required columns:
        ``sample_id``, ``path``.  Any additional columns become
        ``adata.obs`` metadata.
    min_coverage:
        Minimum per-site read coverage (applied during file ingestion).
    max_coverage:
        Maximum per-site read coverage (PCR deduplication filter).
    context:
        Sequence context to retain.  Only relevant for CX_report files.
        Options: ``"CpG"`` (default), ``"CHG"``, ``"CHH"``, ``None`` (all).
    fmt:
        Force a specific reader format: ``"bismark_cov"``, ``"cx_report"``,
        ``"bedgraph"``, or ``"generic"``.  If ``None``, auto-detected from
        file extension.
    join_type:
        How to combine samples across loci: ``"outer"`` (retain all
        detected loci across all samples; NaN for missing sites) or
        ``"inner"`` (retain only loci covered in every sample).
    n_workers:
        Number of threads for parallel file loading.  Defaults to
        ``min(len(samples), cpu_count)``.
    reader_kwargs:
        Additional keyword arguments forwarded to the underlying reader
        function.

    Returns
    -------
    anndata.AnnData
        Cohort-level AnnData object:
          - ``X``                         : beta-value matrix (n_samples × n_sites)
          - ``obs``                       : sample metadata from sample sheet
          - ``var``                       : site coordinates + context
          - ``layers['coverage']``        : total read coverage
          - ``layers['methylated_counts']``: methylated read counts

    Raises
    ------
    FileNotFoundError
        If the sample sheet or any sample file cannot be found.
    ValueError
        If required columns are missing from the sample sheet.

    Examples
    --------
    >>> adata = read_samples("cohort.csv", min_coverage=5, context="CpG")
    >>> print(adata)  # AnnData object with n_obs × n_vars
    """
    import os

    rk = reader_kwargs or {}
    sample_sheet = Path(sample_sheet)
    if not sample_sheet.exists():
        raise FileNotFoundError(f"Sample sheet not found: {sample_sheet}")

    # --- Parse sample sheet ---
    ss = pd.read_csv(sample_sheet)
    required = {"sample_id", "path"}
    missing = required - set(ss.columns)
    if missing:
        raise ValueError(
            f"Sample sheet missing required columns: {missing}. "
            f"Found: {list(ss.columns)}"
        )

    # Validate that all sample files exist
    for _, row in ss.iterrows():
        p = Path(row["path"])
        if not p.exists():
            raise FileNotFoundError(
                f"Sample file not found for '{row['sample_id']}': {p}"
            )

    n_samples = len(ss)
    if n_workers is None:
        n_workers = min(n_samples, os.cpu_count() or 4)

    logger.info("Loading %d samples with %d workers …", n_samples, n_workers)

    # --- Parallel sample loading ---
    results: dict[str, pl.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _load_single_sample,
                row["sample_id"],
                Path(row["path"]),
                fmt or _detect_format(Path(row["path"])),
                min_coverage,
                max_coverage,
                context,
                rk,
            ): row["sample_id"]
            for _, row in ss.iterrows()
        }

        for future in as_completed(futures):
            sid = futures[future]
            try:
                sample_id, df = future.result()
                results[sample_id] = df
                logger.info("  ✓ %s  (%d sites)", sample_id, len(df))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load sample '{sid}': {exc}"
                ) from exc

    # --- Route to appropriate engine ---
    if engine == "duckdb":
        from epykit.io.anndata_builder_duckdb import build_anndata_streaming
        
        # DuckDB engine reads directly from files (doesn't use loaded results)
        adata = build_anndata_streaming(
            sample_ids=list(ss["sample_id"]),
            file_paths=[Path(row["path"]) for _, row in ss.iterrows()],
            obs_metadata=ss[[c for c in ss.columns if c != "path"]].set_index("sample_id"),
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            join_type=join_type,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_threads=duckdb_threads,
        )
    elif engine == "polars":
        # Build sample metadata (obs)
        meta_cols = [c for c in ss.columns if c != "path"]
        obs_df = ss[meta_cols].set_index("sample_id")

        # Preserve order from sample sheet
        ordered_ids = list(ss["sample_id"])
        ordered_dfs = [results[sid] for sid in ordered_ids]

        adata = build_anndata(
            sample_ids=ordered_ids,
            dataframes=ordered_dfs,
            obs_metadata=obs_df,
            join_type=join_type,
        )
    else:
        raise ValueError(
            f"Unknown engine: '{engine}'. Choose 'polars' or 'duckdb'."
        )

    # --- Handle output mode ---
    if output == "zarr":
        if out_path is None:
            raise ValueError(
                "out_path is required when output='zarr'"
            )
        out_path = Path(out_path)
        logger.info("Writing AnnData to Zarr: %s", out_path)
        adata.write_zarr(str(out_path), mode="w")
        adata = ad.read_zarr(str(out_path))
    elif output != "memory":
        raise ValueError(
            f"Unknown output format: '{output}'. Choose 'memory' or 'zarr'."
        )

    return adata
