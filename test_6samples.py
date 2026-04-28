#!/usr/bin/env python3
"""Quick test: 6 samples only to validate the streaming UNION optimization."""

import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

def _get_rss_mb() -> float:
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

def log_msg(msg: str) -> None:
    rss = _get_rss_mb()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} | RSS: {rss:.1f} MB")

os.chdir("/workspaces/epykit")
from epykit.io import build_anndata_streaming

log_msg("Starting 6-sample streaming UNION test")

ss = pd.read_csv("samplesheet.csv")
sample_ids = list(ss["sample_id"])
file_paths = [Path(row["path"]) for _, row in ss.iterrows()]
obs_metadata = ss[["group"]].set_index(pd.Index(ss["sample_id"], name="sample_id"))

with open("test_regions_6.bed", "w") as f:
    f.write("chr1\t13283\t50000\n")

log_msg(f"Testing with {len(sample_ids)} samples")
start = time.time()

try:
    adata = build_anndata_streaming(
        sample_ids=sample_ids,
        file_paths=file_paths,
        obs_metadata=obs_metadata,
        min_coverage=10,
        join_type="outer",
        duckdb_memory_limit="13GB",
        duckdb_threads=3,
        regions_bed="test_regions_6.bed",
    )
    elapsed = time.time() - start
    log_msg(f"✓ SUCCESS: {adata.n_obs} samples × {adata.n_vars} sites in {elapsed:.1f}s")
except Exception as e:
    elapsed = time.time() - start
    log_msg(f"✗ FAILED after {elapsed:.1f}s: {e}")
    raise
