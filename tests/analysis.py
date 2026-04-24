"""End-to-end analysis demo script.

This is a script version of ``test/analysis.ipynb``. It runs the same
pipeline using the real Bismark ``.cov.gz`` files in this directory and
produces QC plots and result tables.

Run from the project root, e.g.::

    python -m test.analysis

or::

    python test/analysis.py

The script will create ``results/`` and ``plots/`` subdirectories next to
this file (i.e. under ``test/``).
"""

from __future__ import annotations

from pathlib import Path
import gc
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epykit.io.bismark import read_bismark_coverage
from epykit.io.anndata_builder import build_anndata, save, load
from epykit.core.methyldata import MethylData
from epykit.stats.tests import calculate_diff_meth
from epykit.stats.dmr import merge_dmrs
from epykit.plot import qc as qc_plots


logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Run the full CONTROL vs CD55 demo analysis.

    This closely follows the original notebook cells:

    1. Imports & configuration
    2. Load Bismark coverage files
    3. Build AnnData
    4. Wrap in MethylData
    5. QC (coverage stats & plots)
    6. Filter coverage
    7. Subset to CpG context
    8. Unite (intersection of sites across samples)
    9. Global methylation
    10. PCA
    11. Sample-to-sample correlation
    12. Differential methylation
    13. Top DMCs & volcano plot
    14. DMRs
    15. Save results
    """

    # ------------------------------------------------------------------
    # 1 · Configuration
    # ------------------------------------------------------------------
    # Directory containing this script and the *.cov.gz files
    data_dir = Path(__file__).resolve().parent

    # Output directories (created under test/ by default)
    result_dir = data_dir / "results"
    plot_dir = data_dir / "plots"
    result_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    print("pymethyl imported \N{CHECK MARK}")
    print(f"Data    → {data_dir}")
    print(f"Results → {result_dir}")
    print(f"Plots   → {plot_dir}")

    matplotlib.rcParams["figure.dpi"] = 120

    # ------------------------------------------------------------------
    # 2 · Load Bismark coverage files
    # ------------------------------------------------------------------
    # Each entry: (file_path, sample_id, group, donor, replicate)
    samples = [
        (
            data_dir
            / "GSM9212480_CONTROLDONOR3_REP1_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz",
            "CTRL_D3_R1",
            "control",
            "D3",
            "R1",
        ),
        (
            data_dir
            / "GSM9212482_CD55DONOR3_REP1_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz",
            "CD55_D3_R1",
            "cd55",
            "D3",
            "R1",
        ),
        (
            data_dir
            / "GSM9212484_CONTROLDONOR4_REP1_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz",
            "CTRL_D4_R1",
            "control",
            "D4",
            "R1",
        ),
        (
            data_dir
            / "GSM9212486_CD55DONOR4_REP1_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz",
            "CD55_D4_R1",
            "cd55",
            "D4",
            "R1",
        ),
        (
            data_dir
            / "GSM9212488_CONTROLDONOR3_REP2_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz",
            "CTRL_D3_R2",
            "control",
            "D3",
            "R2",
        ),
        (
            data_dir
            / "GSM9212490_CD55DONOR3_REP2_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz",
            "CD55_D3_R2",
            "cd55",
            "D3",
            "R2",
        ),
    ]

    dfs: dict[str, pd.DataFrame] = {}
    for path, sid, group, donor, rep in samples:
        if not path.exists():
            raise FileNotFoundError(f"Missing Bismark coverage file: {path}")

        df = read_bismark_coverage(path, min_coverage=7)
        dfs[sid] = df

        # ``median`` works for both pandas and polars
        median_cov = df["coverage"].median()
        print(f"{sid:12s}  {len(df):>9,} sites   {median_cov:.0f}× median cov")

    print("\nExample — first 5 rows of CTRL_D3_R1:")
    print(dfs["CTRL_D3_R1"].head())

    # ------------------------------------------------------------------
    # 3 · Build AnnData (samples × sites)
    # ------------------------------------------------------------------
    obs_meta = (
        pd.DataFrame(
            [
                {"sample_id": sid, "group": group, "donor": donor, "replicate": rep}
                for _, sid, group, donor, rep in samples
            ]
        )
        .set_index("sample_id")
    )

    sample_ids = [sid for _, sid, _, _, _ in samples]
    sample_dfs = [dfs[sid] for sid in sample_ids]
    del dfs
    gc.collect()

    assert set(obs_meta.index) == set(sample_ids)

    adata = build_anndata(
        sample_ids=sample_ids,
        dataframes=sample_dfs,
        obs_metadata=obs_meta,
        join_type="outer",  # or "inner" to reduce sites
        sparse=True,  # store beta as CSR sparse
    )

    print(adata)
    print("\nSample metadata:")
    print(adata.obs)

    # Basic sanity checks similar to a test
    assert adata.n_obs == len(samples), "AnnData should have one obs per sample"
    assert adata.n_vars > 0, "AnnData should contain at least one CpG site"

    # ------------------------------------------------------------------
    # 4 · MethylData wrapper
    # ------------------------------------------------------------------
    mdata = MethylData(adata)
    print(mdata)
    print(f"\nSamples : {mdata.n_samples}")
    print(f"Sites   : {mdata.n_sites:,}")

    assert mdata.n_samples == len(samples)
    assert mdata.n_sites > 0

    # ------------------------------------------------------------------
    # 5 · QC — coverage statistics
    # ------------------------------------------------------------------
    cov_stats = mdata.coverage_stats()
    print("Coverage statistics per sample:")
    print(cov_stats)

    fig = qc_plots.coverage_hist(mdata, max_cov=100)
    fig.savefig(plot_dir / "01_coverage_hist.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6 · Filter coverage
    # ------------------------------------------------------------------
    min_cov = 1
    max_cov = 1000  # remove extreme outliers (PCR artefacts)

    mdata_filt = mdata.filter_coverage(min_coverage=min_cov, max_coverage=max_cov)
    print(f"Before filter : {mdata.n_sites:>10,} sites")
    print(f"After  filter : {mdata_filt.n_sites:>10,} sites")
    print(f"Retained      : {mdata_filt.n_sites / mdata.n_sites * 100:.1f} %")

    assert mdata_filt.n_sites <= mdata.n_sites

    # ------------------------------------------------------------------
    # 7 · Subset to CpG context
    # ------------------------------------------------------------------
    if "context" in mdata_filt.sites.columns:
        mdata_cpg = mdata_filt.subset_context("CpG")
        print(f"CpG sites : {mdata_cpg.n_sites:,}")
    else:
        mdata_cpg = mdata_filt
        print("No 'context' column — using all sites as CpG")
        print(f"Sites : {mdata_cpg.n_sites:,}")

    assert mdata_cpg.n_sites > 0

    # ------------------------------------------------------------------
    # 8 · Unite — keep only sites covered in ALL samples
    # ------------------------------------------------------------------
    mdata_united = mdata_cpg.unite(how="intersect")
    print(f"Sites covered in ALL {mdata_united.n_samples} samples: {mdata_united.n_sites:,}")

    assert mdata_united.n_samples == mdata_cpg.n_samples
    assert mdata_united.n_sites > 0

    # ------------------------------------------------------------------
    # 9 · QC — global methylation per sample
    # ------------------------------------------------------------------
    global_meth = mdata_united.global_methylation()
    print("Global methylation (% per sample):")
    print(global_meth)

    fig = qc_plots.methylation_distribution(mdata_united)
    fig.savefig(plot_dir / "02_meth_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 10 · QC — PCA (coloured by group)
    # ------------------------------------------------------------------
    fig = qc_plots.pca(mdata_united, color_by="group")
    fig.savefig(plot_dir / "03_pca.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 11 · QC — sample-to-sample correlation heatmap
    # ------------------------------------------------------------------
    fig = qc_plots.sample_correlation(mdata_united)
    fig.savefig(plot_dir / "04_sample_corr.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 12 · Differential methylation — CONTROL vs CD55
    # ------------------------------------------------------------------
    logging.info("Running differential methylation analysis (CONTROL vs CD55)...")

    results = calculate_diff_meth(
        mdata_united,
        treatment_col="group",
        test="auto",  # auto = GLM (3 replicates/group)
        fdr_method="BH",
        verbose=True,
    )

    print(f"\nTotal sites tested  : {len(results):,}")
    print(f"Significant (q<0.05): {(results.qvalue < 0.05).sum():,}")
    print(f"Significant (q<0.01): {(results.qvalue < 0.01).sum():,}")
    print(results.head(10))

    assert len(results) == mdata_united.n_sites

    # ------------------------------------------------------------------
    # 13 · Top differentially methylated CpGs (DMCs)
    # ------------------------------------------------------------------
    dmcs = results[(results.qvalue < 0.05) & (results.mean_diff.abs() > 10)].copy()
    dmcs = dmcs.sort_values("mean_diff", key=abs, ascending=False)

    print(f"DMCs (q<0.05 & |Δβ|>10%) : {len(dmcs):,}")
    print(dmcs[["chr", "start", "end", "pvalue", "qvalue", "mean_diff"]].head(20))

    # Simple volcano-style scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        "red" if (q < 0.05 and abs(d) > 10) else "steelblue"
        for q, d in zip(results.qvalue, results.mean_diff)
    ]
    ax.scatter(
        results.mean_diff,
        -np.log10(results.pvalue + 1e-300),
        c=colors,
        s=2,
        alpha=0.4,
        rasterized=True,
    )
    ax.axvline(-10, color="grey", lw=0.8, ls="--")
    ax.axvline(+10, color="grey", lw=0.8, ls="--")
    ax.axhline(-np.log10(0.05), color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Mean methylation difference (CD55 − CONTROL, %)")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title("Volcano plot: CD55 vs CONTROL")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_volcano.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 14 · Merge DMCs → Differentially Methylated Regions (DMRs)
    # ------------------------------------------------------------------
    dmrs = merge_dmrs(
        results,
        qvalue_col="qvalue",
        qvalue_cutoff=0.05,
        diff_col="mean_diff",
        min_sites=3,  # at least 3 CpGs per DMR
    )

    print(f"DMRs found : {len(dmrs):,}")
    if len(dmrs):
        print("\nTop 10 DMRs by number of CpGs:")
        print(dmrs.sort_values("n_sites", ascending=False).head(10))

    # Optional DMR summary
    if len(dmrs):
        if "direction" in dmrs.columns:
            print("DMR direction breakdown:")
            print(dmrs.direction.value_counts())
        print(dmrs.head())

    # ------------------------------------------------------------------
    # 15 · Save results
    # ------------------------------------------------------------------
    save(mdata_united.adata, result_dir / "methyldata_united.h5ad")
    print("Saved methyldata_united.h5ad")

    results.to_csv(result_dir / "diff_meth.tsv", sep="\t", index=False)
    print("Saved diff_meth.tsv")

    dmcs.to_csv(result_dir / "dmcs_q05_d10.tsv", sep="\t", index=False)
    print("Saved dmcs_q05_d10.tsv")

    if len(dmrs):
        dmrs.to_csv(result_dir / "dmrs.tsv", sep="\t", index=False)
        print("Saved dmrs.tsv")

    print("\n\N{CHECK MARK} All results saved to", result_dir.resolve())

    # ------------------------------------------------------------------
    # 16 · (Optional) Reload saved data
    # ------------------------------------------------------------------
    # In a future session you can skip the loading/processing steps and
    # jump straight to analysis:
    #
    # mdata_reloaded = MethylData(load(result_dir / "methyldata_united.h5ad"))
    # print(mdata_reloaded)


if __name__ == "__main__":  # pragma: no cover - manual integration script
    main()
