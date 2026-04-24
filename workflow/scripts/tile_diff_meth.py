"""
workflow/scripts/tile_diff_meth.py
====================================
Snakemake script: tile-level differential methylation.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from epykit.core import MethylData
from epykit.intervals import tile_counts
from epykit.io import load, save
from epykit.stats import calculate_diff_meth

anndata_path  = snakemake.input.anndata          # noqa: F821
out_tiled     = snakemake.output.tiled_anndata   # noqa: F821
out_tsv       = snakemake.output.tiled_dmc       # noqa: F821
window        = snakemake.params.window          # noqa: F821
step          = snakemake.params.step            # noqa: F821
treatment_col = snakemake.params.treatment_col   # noqa: F821

adata = load(anndata_path)

# Tile the genome
tiled = tile_counts(adata, window=window, step=step)
save(tiled, out_tiled, format="h5ad")
print(f"Tiled: {tiled.n_obs} samples × {tiled.n_vars} windows")

# Differential methylation on tiles
mdata_tiled = MethylData(tiled)
mdata_tiled = mdata_tiled.unite(type="intersect")

results = calculate_diff_meth(
    mdata_tiled,
    treatment_col=treatment_col,
    test="auto",
)

results.to_csv(out_tsv, sep="\t", index=False)
print(f"Tiled DMC results written to: {out_tsv}")
