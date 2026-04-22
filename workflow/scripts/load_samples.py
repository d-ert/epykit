"""
workflow/scripts/load_samples.py
=================================
Snakemake script: loads all samples from the sample sheet,
builds the cohort AnnData, and saves QC statistics.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

import pymethyl
from pymethyl.core import MethylData
from pymethyl.io import read_samples, save

# Snakemake-injected variables
sample_sheet  = snakemake.input.sample_sheet        # noqa: F821
out_anndata   = snakemake.output.anndata            # noqa: F821
out_cov_stats = snakemake.output.coverage_stats     # noqa: F821
min_coverage  = snakemake.params.min_coverage       # noqa: F821
max_coverage  = snakemake.params.get("max_coverage", None)  # noqa: F821
context       = snakemake.params.get("context", None)       # noqa: F821

# --- Load cohort ---
adata = read_samples(
    sample_sheet,
    min_coverage=min_coverage,
    max_coverage=max_coverage,
    context=context,
)

# --- QC: coverage stats ---
mdata = MethylData(adata)
cov_stats = mdata.coverage_stats()
cov_stats.to_csv(out_cov_stats, sep="\t")

# --- Save AnnData ---
save(adata, out_anndata, format="h5ad")

print(f"Loaded {adata.n_obs} samples × {adata.n_vars} sites.")
print(f"Saved to: {out_anndata}")
