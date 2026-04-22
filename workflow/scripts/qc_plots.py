"""
workflow/scripts/qc_plots.py
=============================
Snakemake script: generates PCA and sample correlation heatmap.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from pymethyl.core import MethylData
from pymethyl.io import load
from pymethyl import plot

anndata_path  = snakemake.input.anndata         # noqa: F821
out_pca       = snakemake.output.pca_plot       # noqa: F821
out_corr      = snakemake.output.corr_plot      # noqa: F821
treatment_col = snakemake.params.treatment_col  # noqa: F821

adata = load(anndata_path)

# PCA
fig_pca = plot.pca(adata, color_by=treatment_col)
fig_pca.savefig(out_pca, dpi=150, bbox_inches="tight")
print(f"Saved PCA to: {out_pca}")

# Sample correlation heatmap
fig_corr = plot.sample_correlation(adata, color_by=treatment_col)
fig_corr.savefig(out_corr, dpi=150, bbox_inches="tight")
print(f"Saved correlation heatmap to: {out_corr}")
