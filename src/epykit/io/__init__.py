"""
epykit.io — Data ingestion layer
===================================
All reader functions that parse aligner output files into Polars DataFrames
and subsequently construct cohort-level AnnData objects.

Supported formats
-----------------
Bismark  : coverage2cytosine / bismark2bedGraph output  (.cov / .bismark.cov.gz)
Bismark  : CX_report (all cytosine contexts)
Generic  : any tab-separated file with chr/start/end/beta/coverage columns
bedGraph : 4-column bedGraph (chr, start, end, value)

Usage
-----
>>> from epykit.io import read_bismark_coverage, read_samples
>>> df   = read_bismark_coverage("sample.bismark.cov.gz", min_coverage=5)
>>> adata = read_samples("sample_sheet.csv", min_coverage=5)
"""

from epykit.io.anndata_builder import build_anndata, load, save
from epykit.io.anndata_builder_chunked import build_anndata_chunked
from epykit.io.anndata_builder_duckdb import (
    build_anndata_streaming,
    read_samples_streaming,
)
from epykit.io.bismark import read_bismark_coverage, read_bismark_cx_report
from epykit.io.generic import read_bedgraph, read_generic_methylation
from epykit.io.parquet_converter import (
    convert_sample,
    convert_sample_sheet,
)
from epykit.io.sample_sheet import read_samples, read_samples_to_parquet

__all__ = [
    # Bismark
    "read_bismark_coverage",
    "read_bismark_cx_report",
    # Generic / bedGraph
    "read_bedgraph",
    "read_generic_methylation",
    # Multi-sample
    "read_samples",
    "read_samples_to_parquet",
    # Parquet conversion (new)
    "convert_sample",
    "convert_sample_sheet",
    # DuckDB streaming (memory-efficient for large cohorts)
    "build_anndata_streaming",
    "read_samples_streaming",
    # AnnData construction & persistence
    "build_anndata",
    "build_anndata_chunked",
    "save",
    "load",
]
