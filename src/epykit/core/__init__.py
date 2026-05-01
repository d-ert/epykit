"""
epykit.core — Core data structures
======================================
Provides the ``MethylData`` class: an ergonomic, typed wrapper around
``anndata.AnnData`` that exposes the methylation-specific API.

Usage
-----
>>> from epykit.core import MethylData
>>> mdata = MethylData(adata)
>>> mdata = mdata.filter_coverage(5, 500).subset_context("CpG").unite()
"""

from epykit.core.methyldata import MethylData
from epykit.core.parquet_backend import ParquetMethylStore

__all__ = ["MethylData", "ParquetMethylStore"]
