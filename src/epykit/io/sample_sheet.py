"""
epykit.io.sample_sheet
========================
Multi-sample loader: reads a CSV sample sheet, parallelises per-sample
file ingestion using Polars native threading, outer-joins on genomic loci,
and constructs a cohort-level AnnData object.

This module also exposes a Parquet-native conversion entry point for the
new storage model.

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
from typing import TYPE_CHECKING, Union
import shutil

import pandas as pd
import polars as pl

from epykit.io.anndata_builder import build_anndata
from epykit.io.bismark import read_bismark_coverage, read_bismark_cx_report
from epykit.io.generic import read_bedgraph, read_generic_methylation
from epykit.io.parquet_converter import convert_sample_sheet
from epykit.io.regions import load_and_merge_regions

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData
else:  # pragma: no cover
    AnnData = object


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


def _ensure_var_index_safe(adata: "AnnData") -> None:
    """Ensure var.index name does not collide with an existing column.

    AnnData/Zarr writers can behave unpredictably if the index name matches
    a column name with non-identical values. We defensively rename the index
    to avoid collisions.
    """
    index_name = adata.var.index.name
    if index_name and index_name in adata.var.columns:
        new_name = f"{index_name}_index"
        logger.warning(
            "Renaming var.index from '%s' to '%s' to avoid collision with column",
            index_name,
            new_name,
        )
        adata.var.index = adata.var.index.set_names(new_name)


def read_samples_to_parquet(
    sample_sheet: PathLike,
    output_dir: PathLike,
    *,
    n_workers: int | None = None,
    min_coverage: int = 1,
    max_coverage: int | None = None,
    context: str | None = None,
    fmt: str | None = None,
    reader_kwargs: dict | None = None,
    chunksize: int = 2_000_000,
    compression: str = "zstd",
) -> None:
    """Convert a cohort sample sheet into the partitioned Parquet store.

    Parameters
    ----------
    sample_sheet:
        Path to the sample sheet CSV.
    output_dir:
        Root output directory for the Parquet store.
    n_workers:
        Number of worker processes used for per-sample conversion.
    min_coverage, max_coverage, context, fmt:
        Passed through to the underlying converters.
    reader_kwargs:
        Reserved for future format-specific conversion options.
    chunksize:
        Parquet row-group size used when writing each sample.
    compression:
        Parquet compression codec.
    """
    if reader_kwargs:
        logger.debug(
            "reader_kwargs was provided to read_samples_to_parquet but is not used yet: %s",
            sorted(reader_kwargs.keys()),
        )

    sample_sheet = Path(sample_sheet)
    if not sample_sheet.exists():
        raise FileNotFoundError(f"Sample sheet not found: {sample_sheet}")

    logger.info("Converting sample sheet %s to Parquet store %s", sample_sheet, output_dir)
    convert_sample_sheet(
        sample_sheet,
        output_dir,
        n_workers=n_workers,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        context=context,
        fmt=fmt or "auto",
        chunksize=chunksize,
        compression=compression,
    )


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
    regions_bed: PathLike | None = None,
    validate_output: bool = False,
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
    regions_bed:
        Optional BED file (0-based, half-open) to restrict loci during file
        ingestion. Regions are merged per chromosome before filtering.
    validate_output:
        When ``output="zarr"``, whether to re-read the Zarr store after writing.
        Default ``False`` (skip re-read, saves ~1-2 GB peak RAM).
        Set to ``True`` to validate data integrity (costs ~0.5s extra and
        peak RAM spike during I/O completion).

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

    regions_df = None
    if regions_bed is not None:
        regions_df = load_and_merge_regions(regions_bed)

    # --- Route to appropriate engine (early, before Polars loading) ---
    if engine == "duckdb":
        from epykit.io.anndata_builder_duckdb import build_anndata_streaming
        
        # DuckDB engine reads directly from files (no Polars preload needed)
        logger.info("Loading %d samples with DuckDB streaming engine …", n_samples)
        adata = build_anndata_streaming(
            sample_ids=list(ss["sample_id"]),
            file_paths=[Path(row["path"]) for _, row in ss.iterrows()],
            obs_metadata=ss[[c for c in ss.columns if c != "path"]].set_index("sample_id"),
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            join_type=join_type,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_threads=duckdb_threads,
            regions_bed=regions_bed,
        )
    elif engine == "polars":
        logger.info("Loading %d samples with %d workers …", n_samples, n_workers)

        # --- Parallel sample loading (Polars engine) ---
        if regions_df is not None:
            rk = {**rk, "regions": regions_df}

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

        # Ensure we get "write" semantics across anndata versions:
        # remove any existing Zarr store, then write a fresh one.
        if out_path.exists():
            shutil.rmtree(out_path)

        logger.info("Writing AnnData to Zarr: %s", out_path)

        _ensure_var_index_safe(adata)

        adata.write_zarr(str(out_path))
        
        # Conditionally re-read: skip by default to save RAM during peak I/O
        if validate_output:
            logger.info("Validating Zarr output by re-reading (validate_output=True)")
            import anndata as ad

            adata = ad.read_zarr(str(out_path))
        else:
            logger.debug("Skipping Zarr re-read (validate_output=False)")
    elif output != "memory":
        raise ValueError(
            f"Unknown output format: '{output}'. Choose 'memory' or 'zarr'."
        )

    return adata