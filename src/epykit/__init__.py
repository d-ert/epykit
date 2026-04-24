"""EpyKit
======
A highly scalable Python framework for Whole Genome Bisulfite Sequencing (WGBS)
data analysis, interval algebra, and differential methylation modeling.

Designed as a high-performance Python alternative to R's methylKit, built on:
  - Polars  : Rust-powered lazy tabular processing (predicate pushdown)
  - AnnData : Zarr/HDF5-backed annotated data matrices (cohort-scale)
  - polars-bio : Interval algebra for genomic windows / feature annotation
  - statsmodels : Logistic GLM + LRT + HC0 overdispersion correction

AnnData geometry (critical):
  - adata.X                     : beta-value matrix  (n_samples × n_sites)
  - adata.obs                   : sample metadata
  - adata.var                   : site coordinates (chr, start, end, strand, context)
  - adata.layers['coverage']    : total read coverage
  - adata.layers['methylated_counts'] : methylated read counts

Example
-------
>>> import epykit
>>> adata = epykit.io.read_samples("sample_sheet.csv", min_coverage=5)
>>> mdata = epykit.core.MethylData(adata)
>>> mdata = mdata.filter_coverage(5, 500).subset_context("CpG").unite()
>>> results = epykit.stats.calculate_diff_meth(mdata, treatment_col="group")
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Prefer the project/distribution name from pyproject.toml
    __version__ = version("EpyKit")
except PackageNotFoundError:  # pragma: no cover
    # Fallback to normalized name, and finally to a dev placeholder.
    try:  # pragma: no cover
        __version__ = version("epykit")
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "0.0.0.dev0"

__author__ = "EpyKit contributors"
__license__ = "MIT"

# Lazy sub-module imports — avoids importing heavy deps at top level
from epykit import core, intervals, io, plot, stats

__all__ = [
    "__version__",
    "io",
    "core",
    "intervals",
    "stats",
    "plot",
]
