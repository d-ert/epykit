"""
epykit.plot — Visualization and QC
=======================================
Quality control plots, exploratory data analysis, and dimensionality
reduction visualizations built on the AnnData / scverse ecosystem.

Key functions
-------------
coverage_hist(mdata)
    Per-sample coverage distribution histograms.

pca(adata)
    PCA on the beta-value matrix.  Uses scanpy.pp.pca().

assoc_comp(adata)
    Associate PCA components with sample metadata to detect batch effects.
    Kruskal-Wallis for categorical, Pearson for continuous metadata.

umap(adata)
    UMAP dimensionality reduction via scanpy.tl.umap().

sample_correlation(adata)
    Sample-sample Pearson correlation heatmap (seaborn clustermap).

methylation_distribution(mdata)
    Per-sample beta-value distribution violin / histogram plots.

Usage
-----
>>> from epykit.plot import pca, assoc_comp, sample_correlation
>>> pca(mdata.adata)
>>> assoc_comp(mdata.adata)
>>> sample_correlation(mdata.adata)
"""

from epykit.plot.qc import (
    assoc_comp,
    coverage_hist,
    methylation_distribution,
    pca,
    sample_correlation,
    umap,
)

__all__ = [
    "coverage_hist",
    "pca",
    "assoc_comp",
    "umap",
    "sample_correlation",
    "methylation_distribution",
]
