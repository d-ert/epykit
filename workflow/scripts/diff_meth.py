"""
workflow/scripts/diff_meth.py
==============================
Snakemake script: runs differential methylation analysis.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from epykit.core import MethylData
from epykit.io import load
from epykit.stats import calculate_diff_meth

anndata_path   = snakemake.input.anndata         # noqa: F821
out_tsv        = snakemake.output.dmc_tsv        # noqa: F821
treatment_col  = snakemake.params.treatment_col  # noqa: F821
test           = snakemake.params.get("test", "auto")  # noqa: F821
overdispersion = snakemake.params.get("overdispersion", True)  # noqa: F821
fdr_method     = snakemake.params.get("fdr_method", "BH")  # noqa: F821

adata = load(anndata_path)
mdata = MethylData(adata)

# Filter and unite before testing
mdata = mdata.filter_coverage(
    min_cov=snakemake.params.get("min_coverage", 1)  # noqa: F821
).unite(type="intersect")

print(f"Running differential methylation on {mdata.n_sites} sites, {mdata.n_samples} samples.")

results = calculate_diff_meth(
    mdata,
    treatment_col=treatment_col,
    test=test,
    overdispersion=overdispersion,
    fdr_method=fdr_method,
)

results.to_csv(out_tsv, sep="\t", index=False)
print(f"Wrote {len(results)} DMC results to: {out_tsv}")
print(f"Significant (q<0.05): {(results['qvalue'] < 0.05).sum()}")
