"""
pymethyl.core — Core data structures
======================================
Provides the ``MethylData`` class: an ergonomic, typed wrapper around
``anndata.AnnData`` that exposes the methylation-specific API.

Usage
-----
>>> from pymethyl.core import MethylData
>>> mdata = MethylData(adata)
>>> mdata = mdata.filter_coverage(5, 500).subset_context("CpG").unite()
"""

from pymethyl.core.methyldata import MethylData

__all__ = ["MethylData"]
