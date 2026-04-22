"""
pymethyl.io.generic
===================
Readers for generic methylation file formats (bedGraph, tab-separated).

These readers normalise third-party aligner outputs (bwa-meth, BSBolt,
BSMAP, etc.) into the same Polars DataFrame schema used by the Bismark
reader, enabling seamless integration with the rest of the pipeline.

Standard internal schema
------------------------
    chr          : Utf8    — chromosome name
    start        : Int64   — 1-based start position
    end          : Int64   — end position
    strand       : Utf8    — "+" / "-" / "*"
    beta         : Float64 — methylation percentage (0–100)
    methylated   : Int32   — methylated read count
    unmethylated : Int32   — unmethylated read count
    coverage     : Int32   — total read coverage
    context      : Utf8    — "CpG" / "CHG" / "CHH" (optional, default "CpG")
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# bedGraph reader
# ---------------------------------------------------------------------------

def read_bedgraph(
    path: PathLike,
    *,
    value_col: str = "beta",
    min_coverage: int = 1,
    sep: str = "\t",
    has_header: bool = False,
) -> pl.DataFrame:
    """Read a 4-column bedGraph file.

    The standard bedGraph format has **no header** and four columns::

        chr  start  end  value

    For methylation data, ``value`` is typically the percent methylation
    (0–100) or the fractional methylation (0–1).  Values ≤ 1.0 are
    automatically rescaled to 0–100.

    Parameters
    ----------
    path:
        Path to the bedGraph file (optionally gzip-compressed).
    value_col:
        Name to assign the fourth column.  Default: ``"beta"``.
    min_coverage:
        Not applicable to 4-column bedGraphs (no coverage info).
        Retained for API consistency — ignored unless coverage columns exist.
    sep:
        Field separator.  Default: tab.
    has_header:
        Whether the file has a header row.

    Returns
    -------
    pl.DataFrame
        Columns: ``chr``, ``start``, ``end``, ``beta``.

    Notes
    -----
    If the file has extra columns beyond 4 (e.g. bwa-meth bedGraph with
    coverage), use :func:`read_generic_methylation` instead.

    Examples
    --------
    >>> df = read_bedgraph("sample.bedgraph.gz")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"bedGraph file not found: {path}")

    lf = pl.scan_csv(
        str(path),
        separator=sep,
        has_header=has_header,
        new_columns=["chr", "start", "end", value_col],
    )

    # Rescale fractional (0–1) to percentage (0–100) if needed
    lf = lf.with_columns(
        pl.when(pl.col(value_col) <= 1.0)
        .then(pl.col(value_col) * 100.0)
        .otherwise(pl.col(value_col))
        .alias(value_col)
    )

    return lf.collect()


# ---------------------------------------------------------------------------
# Generic tab-separated methylation reader
# ---------------------------------------------------------------------------

def read_generic_methylation(
    path: PathLike,
    *,
    chr_col: str = "chr",
    start_col: str = "start",
    end_col: str = "end",
    beta_col: str | None = "beta",
    methylated_col: str | None = "methylated",
    unmethylated_col: str | None = "unmethylated",
    coverage_col: str | None = None,
    strand_col: str | None = None,
    context_col: str | None = None,
    sep: str = "\t",
    has_header: bool = True,
    comment_char: str | None = "#",
    min_coverage: int = 1,
    max_coverage: int | None = None,
) -> pl.DataFrame:
    """Read a generic tab-separated methylation file.

    This reader handles outputs from bwa-meth, BSBolt, BSMAP, MethPipe,
    and any other tool that produces per-site methylation tables.

    Column mapping is fully configurable via keyword arguments, allowing
    users to specify which columns in their file correspond to the standard
    internal schema.

    Parameters
    ----------
    path:
        Path to the methylation file.
    chr_col, start_col, end_col:
        Column names for genomic coordinates.
    beta_col:
        Column name for percent methylation.  If ``None``, it will be
        computed as ``methylated / coverage * 100``.
    methylated_col:
        Column name for methylated read counts.
    unmethylated_col:
        Column name for unmethylated read counts.  Either this or
        ``coverage_col`` must be provided when ``beta_col`` is ``None``.
    coverage_col:
        Column name for total coverage.  If provided and ``unmethylated_col``
        is ``None``, unmethylated is computed as ``coverage - methylated``.
    strand_col:
        Column name for strand (``+`` / ``-``).  If ``None``, strand is
        set to ``"*"`` (unstranded).
    context_col:
        Column name for sequence context.  If ``None``, context is set to
        ``"CpG"`` (appropriate for most WGBS pipelines).
    sep:
        Field separator character.
    has_header:
        Whether the file has a header row.
    comment_char:
        Lines starting with this character are skipped.
    min_coverage:
        Minimum total coverage to retain a site.
    max_coverage:
        Maximum total coverage to retain a site.

    Returns
    -------
    pl.DataFrame
        Normalised to the standard internal schema:
        ``chr``, ``start``, ``end``, ``strand``, ``beta``,
        ``methylated``, ``unmethylated``, ``coverage``, ``context``.

    Examples
    --------
    >>> # bwa-meth output (has header, coverage column)
    >>> df = read_generic_methylation(
    ...     "sample_bwameth.txt",
    ...     methylated_col="M",
    ...     coverage_col="Cov",
    ...     strand_col="Strand",
    ... )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Methylation file not found: {path}")

    lf = pl.scan_csv(
        str(path),
        separator=sep,
        has_header=has_header,
        comment_prefix=comment_char,
        infer_schema_length=200,
    )

    # --- Rename coordinate columns to standard names ---
    rename_map: dict[str, str] = {}
    if chr_col != "chr":
        rename_map[chr_col] = "chr"
    if start_col != "start":
        rename_map[start_col] = "start"
    if end_col != "end":
        rename_map[end_col] = "end"
    if rename_map:
        lf = lf.rename(rename_map)

    # --- Strand ---
    if strand_col is not None and strand_col != "strand":
        lf = lf.rename({strand_col: "strand"})
    elif strand_col is None:
        lf = lf.with_columns(pl.lit("*").alias("strand"))

    # --- Context ---
    if context_col is not None and context_col != "context":
        lf = lf.rename({context_col: "context"})
    elif context_col is None:
        lf = lf.with_columns(pl.lit("CpG").alias("context"))

    # --- Coverage arithmetic ---
    if methylated_col and methylated_col != "methylated":
        lf = lf.rename({methylated_col: "methylated"})

    if coverage_col is not None:
        if coverage_col != "coverage":
            lf = lf.rename({coverage_col: "coverage"})
        if unmethylated_col is None:
            lf = lf.with_columns(
                (pl.col("coverage") - pl.col("methylated")).cast(pl.Int32).alias("unmethylated")
            )
        elif unmethylated_col != "unmethylated":
            lf = lf.rename({unmethylated_col: "unmethylated"})
    else:
        if unmethylated_col and unmethylated_col != "unmethylated":
            lf = lf.rename({unmethylated_col: "unmethylated"})
        lf = lf.with_columns(
            (pl.col("methylated") + pl.col("unmethylated")).cast(pl.Int32).alias("coverage")
        )

    # --- Beta (percent methylation) ---
    if beta_col is not None and beta_col != "beta":
        lf = lf.rename({beta_col: "beta"})
    elif beta_col is None:
        lf = lf.with_columns(
            (pl.col("methylated").cast(pl.Float64)
             / pl.col("coverage").cast(pl.Float64) * 100.0)
            .alias("beta")
        )

    # Rescale if fractional
    lf = lf.with_columns(
        pl.when(pl.col("beta") <= 1.0)
        .then(pl.col("beta") * 100.0)
        .otherwise(pl.col("beta"))
        .alias("beta")
    )

    # --- Coverage filters (predicate pushdown) ---
    lf = lf.filter(pl.col("coverage") >= min_coverage)
    if max_coverage is not None:
        lf = lf.filter(pl.col("coverage") <= max_coverage)

    # --- Type normalisation ---
    lf = lf.with_columns(
        pl.col("chr").cast(pl.Utf8),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        pl.col("beta").cast(pl.Float64),
        pl.col("methylated").cast(pl.Int32),
        pl.col("unmethylated").cast(pl.Int32),
        pl.col("coverage").cast(pl.Int32),
    )

    # --- Select standard output columns (keeping any extras last) ---
    std_cols = ["chr", "start", "end", "strand", "beta",
                "methylated", "unmethylated", "coverage", "context"]
    existing = lf.columns
    extra = [c for c in existing if c not in std_cols]
    lf = lf.select(std_cols + extra)

    return lf.collect()
