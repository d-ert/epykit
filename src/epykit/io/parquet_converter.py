"""
epykit.io.parquet_converter
============================
Convert methylation data from various formats (Bismark, bedGraph, etc.)
to a partitioned Parquet dataset.

Unified schema
--------------
Each row represents a single CpG site for a single sample.

Column    Type     Description
-------   -------  -----------
chrom     Utf8     Chromosome name (e.g., "chr1")
pos       Int32    1-based position of the CpG start (equivalent to Bismark "start" + 1)
strand    Utf8     "+" or "-" (if available, else "*")
N_meth    Int32    Number of methylated reads
N_unmeth  Int32    Number of unmethylated reads
coverage  Int32    Total coverage (N_meth + N_unmeth)
sample    Utf8     Sample identifier

Partitioning
------------
Files are partitioned by sample and chrom:

    output_dir/
      sample=S1/
        chrom=chr1/
          part-0.parquet
          part-1.parquet
          ...
        chrom=chr2/
          part-0.parquet
      sample=S2/
        ...

This enables efficient per-chromosome and per-sample filtering via Polars' lazy scanning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Union

import polars as pl

from epykit.io.bismark import read_bismark_coverage, read_bismark_cx_report
from epykit.io.generic import read_bedgraph, read_generic_methylation

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Supported input formats
InputFormat = Literal["auto", "bismark_cov", "cx_report", "bedgraph", "generic"]


def _detect_format(path: Path) -> InputFormat:
    """Auto-detect file format from extension."""
    name = path.name.lower()
    if "cx_report" in name:
        return "cx_report"
    suffixes = "".join(path.suffixes).lower()
    if ".bismark.cov" in suffixes or suffixes.endswith(".cov"):
        return "bismark_cov"
    if ".bedgraph" in suffixes or ".bdg" in suffixes:
        return "bedgraph"
    return "generic"


def _load_sample_lazy(
    input_path: PathLike,
    fmt: InputFormat,
    min_coverage: int = 1,
    max_coverage: int | None = None,
    context: str | None = None,
) -> pl.LazyFrame:
    """
    Load a methylation file format and return a lazy frame with standard columns.
    
    Parameters
    ----------
    input_path : PathLike
        Path to the methylation input file.
    fmt : InputFormat
        Format of the input file: "bismark_cov", "cx_report", "bedgraph", or "generic".
        If "auto", attempts to detect from file extension.
    min_coverage : int
        Minimum coverage filter (applied at read time via predicate pushdown).
    max_coverage : int | None
        Maximum coverage filter.
    context : str | None
        Context filter (for CX_report: "CpG", "CHG", "CHH"). None keeps all.
    
    Returns
    -------
    pl.LazyFrame
        Lazy frame with columns: [chr, start, end, strand, methylated, unmethylated, coverage, beta, context]
    """
    input_path = Path(input_path)
    
    if fmt == "auto":
        fmt = _detect_format(input_path)
    
    logger.debug(f"Loading {input_path.name} as format={fmt}")
    
    if fmt == "bismark_cov":
        df = read_bismark_coverage(
            input_path,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            context=context,
        )
        # Add strand column (Bismark .cov doesn't have strand)
        if "strand" not in df.columns:
            df = df.with_columns(pl.lit("*").alias("strand"))
        # Add context column (Bismark .cov = CpG only by default)
        if "context" not in df.columns:
            df = df.with_columns(pl.lit("CpG").alias("context"))
        return df.lazy()
    
    elif fmt == "cx_report":
        df = read_bismark_cx_report(
            input_path,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            context=context,
        )
        return df.lazy()
    
    elif fmt == "bedgraph":
        df = read_bedgraph(
            input_path,
            value_col="beta",
            min_coverage=min_coverage,
        )
        # Add missing columns for standardization
        for col in ["strand", "methylated", "unmethylated", "coverage", "context"]:
            if col not in df.columns:
                if col == "strand":
                    df = df.with_columns(pl.lit("*").alias(col))
                elif col == "context":
                    df = df.with_columns(pl.lit("CpG").alias(col))
                else:
                    # bedGraph typically cannot infer methylated/unmethylated; use 0 as placeholder
                    df = df.with_columns(pl.lit(0).cast(pl.Int32).alias(col))
        return df.lazy()
    
    elif fmt == "generic":
        df = read_generic_methylation(
            input_path,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        )
        # Add missing columns
        for col in ["strand", "context"]:
            if col not in df.columns:
                if col == "strand":
                    df = df.with_columns(pl.lit("*").alias(col))
                elif col == "context":
                    df = df.with_columns(pl.lit("CpG").alias(col))
        return df.lazy()
    
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _normalize_to_unified_schema(lf: pl.LazyFrame, sample_name: str) -> pl.LazyFrame:
    """
    Transform the standard columns to the unified Parquet schema.
    
    Expected input columns: chr, start, end, strand, methylated, unmethylated, coverage, beta
    
    Output columns: chrom, pos, strand, N_meth, N_unmeth, coverage, sample
    
    Notes
    -----
    - Converts 0-based start (from Bismark, bedGraph) to 1-based pos.
    - Renames columns to match unified schema.
    - Adds sample identifier.
    - Ensures correct data types.
    """
    lf = lf.with_columns([
        pl.col("chr").alias("chrom"),
        (pl.col("start") + 1).cast(pl.Int32).alias("pos"),  # 0-based -> 1-based
        pl.col("strand").cast(pl.Utf8),
        pl.col("methylated").cast(pl.Int32).alias("N_meth"),
        pl.col("unmethylated").cast(pl.Int32).alias("N_unmeth"),
        pl.col("coverage").cast(pl.Int32),
        pl.lit(sample_name).alias("sample"),
    ]).select([
        "chrom", "pos", "strand", "N_meth", "N_unmeth", "coverage", "sample"
    ])
    
    return lf


def convert_sample(
    input_path: PathLike,
    sample_name: str,
    output_dir: PathLike,
    *,
    fmt: InputFormat = "auto",
    min_coverage: int = 1,
    max_coverage: int | None = None,
    context: str | None = None,
    chunksize: int = 2_000_000,
    compression: str = "zstd",
) -> None:
    """
    Convert a single methylation sample file to partitioned Parquet.
    
    Parameters
    ----------
    input_path : PathLike
        Path to the input methylation file (Bismark, bedGraph, etc.).
    sample_name : str
        Unique sample identifier (must be filesystem-safe: alphanumeric, _, -).
        This becomes the sample partition key.
    output_dir : PathLike
        Root directory where the Parquet dataset will be written.
        Creates/appends to: output_dir/sample={sample_name}/chrom={chrom}/part-*.parquet
    fmt : InputFormat
        Input format ("auto", "bismark_cov", "cx_report", "bedgraph", "generic").
        If "auto", detects from file extension.
    min_coverage : int
        Minimum coverage filter. Default: 1 (keep all).
    max_coverage : int | None
        Maximum coverage filter (e.g., for PCR duplicate removal). Default: None.
    context : str | None
        Context filter ("CpG", "CHG", "CHH"). None keeps all. Default: None.
    chunksize : int
        Approximate row group size for Parquet (rows per chunk). Larger values
        increase memory during write but improve compression. Default: 2_000_000.
    compression : str
        Parquet compression codec: "zstd", "snappy", "gzip", etc. Default: "zstd".
    
    Returns
    -------
    None
        Writes partitioned Parquet dataset to output_dir.
    
    Notes
    -----
    All filtering (coverage, context) is applied lazily via predicate pushdown
    before any data enters RAM, minimizing memory usage.
    
    The Parquet write uses streaming (sink_parquet) to avoid loading the full
    sample into memory at once.
    
    Examples
    --------
    >>> from epykit.io.parquet_converter import convert_sample
    >>> convert_sample(
    ...     "sample1.bismark.cov.gz",
    ...     "S1",
    ...     "methylstore",
    ...     min_coverage=10,
    ... )
    >>> # Creates: methylstore/sample=S1/chrom=chr1/part-*.parquet, etc.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Validate sample name
    if not all(c.isalnum() or c in "_-" for c in sample_name):
        raise ValueError(
            f"Sample name '{sample_name}' contains invalid characters. "
            f"Use only alphanumeric, underscore, and hyphen."
        )
    
    logger.info(f"Converting {input_path.name} (sample={sample_name})")
    
    # Load and normalize to unified schema
    lf = _load_sample_lazy(
        input_path,
        fmt=fmt,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        context=context,
    )
    lf = _normalize_to_unified_schema(lf, sample_name)
    
    try:
        chrom_names = (
            lf.select(pl.col("chrom").unique().sort())
            .collect()
            .get_column("chrom")
            .to_list()
        )

        logger.debug(
            "Writing %d chromosome partitions for sample=%s to %s",
            len(chrom_names),
            sample_name,
            output_dir,
        )

        for chrom in chrom_names:
            chrom_dir = output_dir / f"sample={sample_name}" / f"chrom={chrom}"
            chrom_dir.mkdir(parents=True, exist_ok=True)
            out_file = chrom_dir / "part-0.parquet"

            chrom_lf = lf.filter(pl.col("chrom") == chrom)
            chrom_lf.sink_parquet(
                str(out_file),
                compression=compression,
                row_group_size=chunksize,
                maintain_order=False,
            )

        logger.info(f"✓ Successfully wrote sample {sample_name} to {output_dir}")
    except Exception as e:
        logger.error(f"✗ Failed to convert {sample_name}: {e}")
        raise


def convert_sample_sheet(
    sample_sheet_path: PathLike,
    output_dir: PathLike,
    *,
    path_col: str = "path",
    sample_id_col: str = "sample_id",
    fmt: InputFormat | None = None,
    n_workers: int | None = None,
    **kwargs,
) -> None:
    """
    Convert all samples in a CSV sample sheet to partitioned Parquet in parallel.
    
    Parameters
    ----------
    sample_sheet_path : PathLike
        Path to CSV with columns: sample_id, path, [fmt], [group], [...].
        Minimal columns: sample_id, path.
    output_dir : PathLike
        Root output directory for the Parquet dataset.
    path_col : str
        Column name containing file paths. Default: "path".
    sample_id_col : str
        Column name containing sample identifiers. Default: "sample_id".
    fmt : InputFormat | None
        Optional global input format override. If None, the function uses a
        per-sample "fmt" column when available, otherwise auto-detects from
        file extension.
    n_workers : int | None
        Number of parallel workers (ProcessPoolExecutor). If None, uses CPU count.
    **kwargs
        Additional arguments passed to convert_sample() (min_coverage, max_coverage, context, etc.).
    
    Returns
    -------
    None
    
    Notes
    -----
    The sample sheet can optionally include a "fmt" column to specify format per sample.
    If not present, format is auto-detected from file extension.
    
    Examples
    --------
    >>> from epykit.io.parquet_converter import convert_sample_sheet
    >>> convert_sample_sheet(
    ...     "sample_sheet.csv",
    ...     "methylstore",
    ...     min_coverage=10,
    ...     n_workers=4,
    ... )
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    sample_sheet = pd.read_csv(sample_sheet_path)
    
    required_cols = {sample_id_col, path_col}
    if not required_cols.issubset(sample_sheet.columns):
        raise ValueError(
            f"Sample sheet missing required columns: {required_cols}. "
            f"Found: {set(sample_sheet.columns)}"
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if fmt column exists
    has_fmt_col = "fmt" in sample_sheet.columns
    
    if n_workers is None:
        import os
        n_workers = os.cpu_count() or 1
    
    logger.info(
        f"Converting {len(sample_sheet)} samples with {n_workers} workers "
        f"to {output_dir}"
    )
    
    tasks = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, row in sample_sheet.iterrows():
            sample_id = row[sample_id_col]
            file_path = row[path_col]
            sample_fmt = fmt if fmt is not None else (
                row.get("fmt", "auto") if has_fmt_col else "auto"
            )
            
            task = executor.submit(
                convert_sample,
                file_path,
                sample_id,
                output_dir,
                fmt=sample_fmt,
                **kwargs,
            )
            tasks.append((sample_id, task))
        
        # Collect results
        completed = 0
        failed = 0
        for sample_id, task in tasks:
            try:
                task.result()
                completed += 1
            except Exception as e:
                logger.error(f"Failed to convert {sample_id}: {e}")
                failed += 1
        
        logger.info(
            f"Conversion complete: {completed} succeeded, {failed} failed"
        )
        
        if failed > 0:
            raise RuntimeError(
                f"{failed} samples failed to convert. See logs for details."
            )
