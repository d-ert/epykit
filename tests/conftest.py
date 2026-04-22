"""
tests/conftest.py
=================
Shared pytest fixtures for py-methyl-toolkit tests.

Toy 5-sample / 10-site dataset for fast unit tests.
All fixtures return deterministic data for reproducibility.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Toy methylation data
# ---------------------------------------------------------------------------

SITES = [
    ("chr1", 1000,  1001, "+"),
    ("chr1", 2000,  2001, "+"),
    ("chr1", 3000,  3001, "-"),
    ("chr1", 5000,  5001, "+"),
    ("chr1", 8000,  8001, "+"),
    ("chr2", 500,   501,  "+"),
    ("chr2", 1500,  1501, "+"),
    ("chr2", 4000,  4001, "-"),
    ("chr3", 100,   101,  "+"),
    ("chr3", 9999,  10000, "+"),
]

N_SITES = len(SITES)
N_SAMPLES = 4

# Reproducible RNG
RNG = np.random.default_rng(42)

# Two control + two treated samples
SAMPLE_IDS = ["ctrl_1", "ctrl_2", "treat_1", "treat_2"]
GROUPS = ["control", "control", "treatment", "treatment"]


def _make_beta_matrix() -> np.ndarray:
    """Create a deterministic beta matrix with some differential sites."""
    beta = RNG.uniform(20, 80, (N_SAMPLES, N_SITES)).astype(np.float32)
    # Make sites 0-2 differentially methylated
    beta[0, :3] = [30.0, 35.0, 28.0]   # ctrl_1
    beta[1, :3] = [32.0, 33.0, 31.0]   # ctrl_2
    beta[2, :3] = [70.0, 75.0, 72.0]   # treat_1
    beta[3, :3] = [68.0, 73.0, 74.0]   # treat_2
    return beta


def _make_coverage_matrix() -> np.ndarray:
    """Create a deterministic coverage matrix."""
    cov = RNG.integers(10, 50, (N_SAMPLES, N_SITES)).astype(np.int32)
    return cov


@pytest.fixture(scope="session")
def toy_adata() -> ad.AnnData:
    """Return a toy AnnData object (4 samples × 10 sites)."""
    beta = _make_beta_matrix()
    cov = _make_coverage_matrix()

    # Compute methylated counts from beta and coverage
    meth = np.round(beta / 100.0 * cov).astype(np.int32)
    meth = np.clip(meth, 0, cov)

    # Build var DataFrame
    locus_keys = [f"{c}:{s}-{e}:{strand}" for c, s, e, strand in SITES]
    var_df = pd.DataFrame(
        [{"chr": c, "start": s, "end": e, "strand": strand, "context": "CpG"}
         for c, s, e, strand in SITES],
        index=locus_keys,
    )
    var_df.index.name = "locus_key"

    # Build obs DataFrame
    obs_df = pd.DataFrame(
        {"group": GROUPS, "batch": ["A", "A", "B", "B"]},
        index=SAMPLE_IDS,
    )
    obs_df.index.name = "sample_id"

    adata = ad.AnnData(
        X=beta,
        obs=obs_df,
        var=var_df,
        layers={
            "coverage": cov,
            "methylated_counts": meth,
        },
    )
    return adata


@pytest.fixture
def toy_mdata(toy_adata):
    """Return a MethylData wrapper around the toy AnnData."""
    from pymethyl.core import MethylData
    return MethylData(toy_adata.copy())


# ---------------------------------------------------------------------------
# Bismark coverage file fixtures
# ---------------------------------------------------------------------------

BISMARK_COV_CONTENT = """\
chr1\t1000\t1001\t30.0\t30\t70
chr1\t2000\t2001\t35.0\t35\t65
chr1\t3000\t3001\t28.0\t14\t36
chr1\t5000\t5001\t60.0\t30\t20
chr1\t8000\t8001\t75.0\t45\t15
chr2\t500\t501\t50.0\t25\t25
chr2\t1500\t1501\t20.0\t4\t16
chr2\t4000\t4001\t80.0\t32\t8
chr3\t100\t101\t5.0\t1\t19
chr3\t9999\t10000\t90.0\t9\t1
"""

BISMARK_COV_LOW_COV = """\
chr1\t1000\t1001\t30.0\t3\t7
chr1\t2000\t2001\t35.0\t35\t65
chr1\t3000\t3001\t28.0\t14\t36
"""


@pytest.fixture
def bismark_cov_file(tmp_path) -> Path:
    """Write toy Bismark coverage content to a temporary file."""
    f = tmp_path / "sample.bismark.cov"
    f.write_text(BISMARK_COV_CONTENT)
    return f


@pytest.fixture
def bismark_low_cov_file(tmp_path) -> Path:
    """Bismark coverage file with one low-coverage site."""
    f = tmp_path / "sample_lowcov.bismark.cov"
    f.write_text(BISMARK_COV_LOW_COV)
    return f


# ---------------------------------------------------------------------------
# Sample sheet fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sheet_dir(tmp_path) -> Path:
    """Create 2 temporary Bismark files and a matching sample sheet."""
    # Write sample files
    for i in range(1, 3):
        sample_file = tmp_path / f"sample{i}.bismark.cov"
        sample_file.write_text(BISMARK_COV_CONTENT)

    # Write sample sheet
    sheet = tmp_path / "sample_sheet.csv"
    sheet.write_text(
        "sample_id,path,group,batch\n"
        f"ctrl_1,{tmp_path / 'sample1.bismark.cov'},control,A\n"
        f"treat_1,{tmp_path / 'sample2.bismark.cov'},treatment,B\n"
    )
    return tmp_path
