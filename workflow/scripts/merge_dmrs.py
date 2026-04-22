"""
workflow/scripts/merge_dmrs.py
================================
Snakemake script: merges adjacent significant DMCs into DMRs.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

import pandas as pd
from pymethyl.stats import merge_dmrs
from pymethyl.stats.dmr import dmrs_to_bed

dmc_tsv   = snakemake.input.dmc_tsv     # noqa: F821
out_bed   = snakemake.output.dmr_bed    # noqa: F821
out_tsv   = snakemake.output.dmr_tsv    # noqa: F821
max_gap   = snakemake.params.max_gap    # noqa: F821
min_sites = snakemake.params.min_sites  # noqa: F821

dmc = pd.read_csv(dmc_tsv, sep="\t")
dmrs = merge_dmrs(
    dmc,
    max_gap=max_gap,
    min_sites=min_sites,
    qvalue_cutoff=0.05,
    min_abs_diff=10.0,
)

dmrs.to_csv(out_tsv, sep="\t", index=False)
dmrs_to_bed(dmrs, out_bed)

print(f"Found {len(dmrs)} DMRs.")
print(f"  Hyper: {(dmrs['direction'] == 'hyper').sum()}")
print(f"  Hypo:  {(dmrs['direction'] == 'hypo').sum()}")
