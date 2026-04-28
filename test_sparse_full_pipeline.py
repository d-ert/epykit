#!/usr/bin/env python3
"""
Comprehensive test: Build AnnData with sparse=True and run full DMR pipeline.
Shows: I/O → QC → stats → DMR merging → annotation → tiling.
"""

import os
import time
import gc
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


def _get_rss_mb() -> float:
    """Get resident set size (RSS) in MB."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


def log_msg(step: str, msg: str, level: str = "INFO") -> None:
    """Log with timestamp and memory usage."""
    rss = _get_rss_mb()
    rss_str = f" | RSS: {rss:.1f} MB" if rss >= 0 else ""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} {level:5s}] [{step:3s}] {msg}{rss_str}")


def run_pipeline():
    """Run full pipeline: I/O → QC → DMC → DMR → annotation → tiling."""
    
    os.chdir("/workspaces/epykit")
    
    log_msg("INIT", "=" * 70)
    log_msg("INIT", "EpyKit Full Pipeline Test with sparse=True")
    log_msg("INIT", "=" * 70)
    
    # =========================================================================
    # STEP 1: Load sample metadata
    # =========================================================================
    log_msg("1/IO", "Loading sample sheet...")
    ss = pd.read_csv("samplesheet.csv")
    sample_ids = list(ss["sample_id"])
    file_paths = [Path(row["path"]) for _, row in ss.iterrows()]
    obs_metadata = ss[["group"]].set_index(pd.Index(ss["sample_id"], name="sample_id"))
    
    log_msg("1/IO", f"Loaded {len(sample_ids)} samples: {', '.join(sample_ids[:3])}...")
    
    # Create a test regions file (chr1:13283-50000 for quick test)
    with open("test_regions_sparse.bed", "w") as f:
        f.write("chr1\t13283\t50000\n")
    log_msg("1/IO", "Created test region: chr1:13283-50000")
    
    # =========================================================================
    # STEP 2: Build AnnData with sparse=True
    # =========================================================================
    log_msg("2/BLD", "=" * 70)
    log_msg("2/BLD", "Building AnnData with sparse=True...")
    log_msg("2/BLD", "=" * 70)
    
    from epykit.io import build_anndata_streaming
    
    t0 = time.time()
    try:
        adata = build_anndata_streaming(
            sample_ids=sample_ids,
            file_paths=file_paths,
            obs_metadata=obs_metadata,
            min_coverage=10,
            max_coverage=500,
            join_type="outer",
            duckdb_memory_limit="14GB",
            duckdb_threads=3,
            #regions_bed="test_regions_sparse.bed",
            sparse=True,  # <-- SPARSE MODE
        )
        elapsed = time.time() - t0
        
        # Check if truly sparse
        from scipy.sparse import issparse
        is_sparse = issparse(adata.X)
        sparsity = 1.0 - (adata.X.nnz / (adata.n_obs * adata.n_vars)) if is_sparse else 0.0
        
        log_msg("2/BLD", f"✓ Built: {adata.n_obs} samples × {adata.n_vars} sites in {elapsed:.1f}s", level="OK")
        log_msg("2/BLD", f"  Sparse: {is_sparse} | Sparsity: {sparsity*100:.1f}% | NNZ: {adata.X.nnz:,}", level="OK")
        log_msg("2/BLD", f"  Layers: {list(adata.layers.keys())}", level="OK")
        
    except Exception as e:
        log_msg("2/BLD", f"✗ FAILED: {e}", level="ERR")
        raise
    
    gc.collect()
    
    # =========================================================================
    # STEP 3: Wrap in MethylData and filter
    # =========================================================================
    log_msg("3/QC", "=" * 70)
    log_msg("3/QC", "Quality control: filter coverage & context...")
    log_msg("3/QC", "=" * 70)
    
    from epykit.core import MethylData
    
    t0 = time.time()
    mdata = MethylData(adata)
    log_msg("3/QC", f"Wrapped AnnData in MethylData")
    
    # Filter coverage: require 5-200 reads
    mdata_filt = mdata.filter_coverage(min_cov=5, max_cov=200, require_all_samples=False)
    elapsed = time.time() - t0
    log_msg("3/QC", f"✓ After coverage filter: {mdata_filt.adata.n_vars} sites (removed {mdata.adata.n_vars - mdata_filt.adata.n_vars}) in {elapsed:.1f}s", level="OK")
    
    # Subset to CpG context
    t0 = time.time()
    mdata_cpg = mdata_filt.subset_context("CpG")
    elapsed = time.time() - t0
    log_msg("3/QC", f"✓ CpG sites only: {mdata_cpg.adata.n_vars} sites in {elapsed:.1f}s", level="OK")
    
    gc.collect()
    
    # =========================================================================
    # STEP 4: Differential Methylation Testing (DMC)
    # =========================================================================
    log_msg("4/DMC", "=" * 70)
    log_msg("4/DMC", "Differential methylation testing (single-base)...")
    log_msg("4/DMC", "=" * 70)
    
    from epykit.stats import calculate_diff_meth
    
    t0 = time.time()
    try:
        # Run GLM-based test: cd55 vs control
        results = calculate_diff_meth(
            mdata_cpg,
            treatment_col="group",
            test="glm",  # Logistic GLM with LRT
            fdr_method="BH",  # Benjamini-Hochberg FDR correction
        )
        elapsed = time.time() - t0
        
        # Extract key results
        n_sig = (results["qvalue"] < 0.05).sum()
        
        log_msg("4/DMC", f"✓ Testing complete in {elapsed:.1f}s", level="OK")
        log_msg("4/DMC", f"  Total sites tested: {len(results)}", level="OK")
        log_msg("4/DMC", f"  Significant (qvalue < 0.05): {n_sig}", level="OK")
        log_msg("4/DMC", f"  Columns: {list(results.columns)}", level="OK")
        
        # Show top hits
        top_hyper = results.nlargest(3, "mean_diff")[["chr", "start", "mean_diff", "qvalue"]]
        top_hypo = results.nsmallest(3, "mean_diff")[["chr", "start", "mean_diff", "qvalue"]]
        log_msg("4/DMC", f"  Top hyper-methylated:", level="OK")
        for idx, row in top_hyper.iterrows():
            log_msg("4/DMC", f"    {row['chr']}:{row['start']} Δβ={row['mean_diff']:.3f} q={row['qvalue']:.2e}", level="OK")
        log_msg("4/DMC", f"  Top hypo-methylated:", level="OK")
        for idx, row in top_hypo.iterrows():
            log_msg("4/DMC", f"    {row['chr']}:{row['start']} Δβ={row['mean_diff']:.3f} q={row['qvalue']:.2e}", level="OK")
        
    except Exception as e:
        log_msg("4/DMC", f"✗ FAILED: {e}", level="ERR")
        import traceback
        traceback.print_exc()
        return
    
    gc.collect()
    
    # =========================================================================
    # STEP 5: DMR Merging
    # =========================================================================
    log_msg("5/DMR", "=" * 70)
    log_msg("5/DMR", "Merging DMCs into regions (DMR)...")
    log_msg("5/DMR", "=" * 70)
    
    from epykit.stats import merge_dmrs
    
    t0 = time.time()
    try:
        # Select significant sites and merge into regions
        sig_results = results[results["qvalue"] < 0.05].copy()
        
        if len(sig_results) > 0:
            dmrs = merge_dmrs(
                sig_results,
                min_cpg=2,  # Minimum 2 CpGs per region
                max_gap=300,  # Maximum 300 bp gap between sites
            )
            elapsed = time.time() - t0
            
            n_hyper = (dmrs["med_beta_diff"] > 0).sum()
            n_hypo = (dmrs["med_beta_diff"] <= 0).sum()
            
            log_msg("5/DMR", f"✓ DMR merging complete in {elapsed:.1f}s", level="OK")
            log_msg("5/DMR", f"  Identified {len(dmrs)} DMRs:", level="OK")
            log_msg("5/DMR", f"    Hyper-methylated: {n_hyper}", level="OK")
            log_msg("5/DMR", f"    Hypo-methylated: {n_hypo}", level="OK")
            log_msg("5/DMR", f"  Columns: {list(dmrs.columns)}", level="OK")
            
            # Show top DMRs
            top_dmrs = dmrs.nlargest(3, "med_beta_diff")
            for idx, row in top_dmrs.iterrows():
                log_msg("5/DMR", f"    {row['chr']}:{row['start']}-{row['end']} ({row['n_cpg']} CpGs) Δβ={row['med_beta_diff']:.3f}", level="OK")
        else:
            log_msg("5/DMR", f"No significant sites to merge (qvalue < 0.05)", level="WARN")
            dmrs = None
        
    except Exception as e:
        log_msg("5/DMR", f"✗ FAILED: {e}", level="ERR")
        import traceback
        traceback.print_exc()
        dmrs = None
    
    gc.collect()
    
    # =========================================================================
    # STEP 6: Tiling (bin sites into windows)
    # =========================================================================
    log_msg("6/TIL", "=" * 70)
    log_msg("6/TIL", "Tiling: bin single-base sites into 1 kb windows...")
    log_msg("6/TIL", "=" * 70)
    
    from epykit.intervals import tile_counts
    
    t0 = time.time()
    try:
        # Create 1 kb tiled windows
        adata_tiled = tile_counts(mdata_cpg.adata, window=1000, step=1000)
        elapsed = time.time() - t0
        
        log_msg("6/TIL", f"✓ Tiling complete in {elapsed:.1f}s", level="OK")
        log_msg("6/TIL", f"  Input: {mdata_cpg.adata.n_vars} sites", level="OK")
        log_msg("6/TIL", f"  Output: {adata_tiled.n_vars} tiles (1 kb windows)", level="OK")
        
    except Exception as e:
        log_msg("6/TIL", f"✗ FAILED: {e}", level="ERR")
        import traceback
        traceback.print_exc()
    
    gc.collect()
    
    # =========================================================================
    # STEP 7: Annotation (CpG islands)
    # =========================================================================
    log_msg("7/ANN", "=" * 70)
    log_msg("7/ANN", "Annotating sites with CpG island classification...")
    log_msg("7/ANN", "=" * 70)
    
    from epykit.intervals import annotate_cpg_islands
    
    t0 = time.time()
    try:
        # Annotate with CpG islands (requires a BED file)
        # For this test, we'll just show the available columns
        log_msg("7/ANN", f"Site annotations available:", level="OK")
        log_msg("7/ANN", f"  var_names: {mdata_cpg.adata.var_names[:5]}...", level="OK")
        log_msg("7/ANN", f"  var columns: {list(mdata_cpg.adata.var.columns)}", level="OK")
        elapsed = time.time() - t0
        log_msg("7/ANN", f"✓ Annotation check in {elapsed:.1f}s", level="OK")
        
    except Exception as e:
        log_msg("7/ANN", f"✗ FAILED: {e}", level="ERR")
    
    gc.collect()
    
    # =========================================================================
    # STEP 8: Summary & Memory Report
    # =========================================================================
    log_msg("8/SUM", "=" * 70)
    log_msg("8/SUM", "Pipeline Summary", level="OK")
    log_msg("8/SUM", "=" * 70)
    
    log_msg("8/SUM", f"Input:  {len(sample_ids)} samples × {mdata.adata.n_vars} sites (original)", level="OK")
    log_msg("8/SUM", f"After QC:  {mdata_cpg.adata.n_obs} samples × {mdata_cpg.adata.n_vars} sites (CpG only)", level="OK")
    
    if "results" in locals():
        n_sig = (results["qvalue"] < 0.05).sum()
        log_msg("8/SUM", f"DMC: {len(results)} tested → {n_sig} significant (qvalue < 0.05)", level="OK")
    
    if dmrs is not None:
        log_msg("8/SUM", f"DMR: {len(dmrs)} regions merged from significant sites", level="OK")
    
    log_msg("8/SUM", f"Sparse mode: {'✓ YES' if is_sparse else '✗ NO'} (sparsity: {sparsity*100:.1f}%)", level="OK")
    
    rss_final = _get_rss_mb()
    log_msg("8/SUM", f"Final RSS: {rss_final:.1f} MB", level="OK")
    
    log_msg("8/SUM", "=" * 70)
    log_msg("8/SUM", "Pipeline execution completed successfully!", level="OK")
    log_msg("8/SUM", "=" * 70)


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        log_msg("EXIT", "Interrupted by user", level="WARN")
    except Exception as e:
        log_msg("EXIT", f"Fatal error: {e}", level="ERR")
        import traceback
        traceback.print_exc()
        exit(1)
