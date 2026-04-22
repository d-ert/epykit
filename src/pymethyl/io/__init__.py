"""
pymethyl.io — Data ingestion layer
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
>>> from pymethyl.io import read_bismark_coverage, read_samples
>>> df   = read_bismark_coverage("sample.bismark.cov.gz", min_coverage=5)
>>> adata = read_samples("sample_sheet.csv", min_coverage=5)
"""

from pymethyl.io.anndata_builder import build_anndata, load, save
from pymethyl.io.bismark import read_bismark_coverage, read_bismark_cx_report
from pymethyl.io.generic import read_bedgraph, read_generic_methylation
from pymethyl.io.sample_sheet import read_samples

__all__ = [
    # Bismark
    "read_bismark_coverage",
    "read_bismark_cx_report",
    # Generic / bedGraph
    "read_bedgraph",
    "read_generic_methylation",
    # Multi-sample
    "read_samples",
    # AnnData construction & persistence
    "build_anndata",
    "save",
    "load",
]
