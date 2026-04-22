"""
pymethyl.intervals — Genomic interval algebra
===============================================
Tiling window operations, feature annotation, and CpG island classification
using polars-bio (primary backend) with PyRanges as fallback.

Key functions
-------------
tile_counts(adata, window, step)
    Bin single-base CpG sites into fixed-size genomic windows, summing
    methylated counts and coverage per window per sample.

annotate_features(adata, bed_file)
    Overlap CpG sites with a BED file of genomic features (promoters,
    gene bodies, enhancers) and store results in adata.var.

annotate_cpg_islands(adata, bed_file)
    Classify CpG sites as island / shore / shelf / open sea.

Usage
-----
>>> from pymethyl.intervals import tile_counts, annotate_features
>>> tiled = tile_counts(mdata.adata, window=1000, step=1000)
>>> adata_ann = annotate_features(mdata.adata, "hg38_promoters.bed")
"""

from pymethyl.intervals.tiling import annotate_cpg_islands, annotate_features, tile_counts

__all__ = [
    "tile_counts",
    "annotate_features",
    "annotate_cpg_islands",
]
