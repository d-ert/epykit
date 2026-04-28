# EpyKit 🧬

A highly scalable, memory-efficient Python framework for **Whole Genome Bisulfite Sequencing (WGBS)** data analysis. Built as a modern alternative to R's `methylKit`, with dramatic RAM savings through DuckDB streaming and sparse matrix support.

## Key Features

- 🚀 **Memory-efficient I/O** — DuckDB streaming engine processes full genomes (42M+ loci) in ~4–6 GB RAM (vs. 17–20 GB)
- 🎯 **Sparse matrix mode** — 50–70% storage reduction for outer-join datasets with incomplete coverage
- 🔬 **Ergonomic analysis** — `MethylData` wrapper mirroring R's methylKit API with typed accessors
- 📏 **Multiple engines** — Polars (fast, in-memory) for small cohorts; DuckDB (streaming) for large datasets
- 📊 **Complete statistics** — Fisher exact test, Logistic GLM + likelihood ratio test, HC0 overdispersion correction, BH-FDR correction, DMR merging
- 📈 **Quality control** — PCA, UMAP, sample correlation heatmaps, coverage diagnostics
- 📐 **Interval operations** — Genomic tiling windows, feature annotation, CpG island classification

## Supported Aligners & File Formats

| Aligner | Format | Reader | Status |
|---------|--------|--------|--------|
| Bismark | `.bismark.cov.gz` / `.cov` | `read_bismark_coverage()` | ✅ Fully working |
| Bismark | `CX_report_[CpG/CHG/CHH]` | `read_bismark_cx_report()` | ✅ Fully working |
| bwa-meth | bedGraph | `read_bedgraph()` | ✅ Fully working |
| Generic | Tab-separated | `read_generic_methylation()` | ✅ Fully working |

## Quick Start

### **Option 1: Polars Engine (Fast, In-Memory)**
Best for small to medium cohorts (<50 samples, <10M sites per sample).

```python
import epykit
from epykit.core import MethylData

# 1. Load samples from CSV sheet
adata = epykit.io.read_samples(
    "sample_sheet.csv",
    min_coverage=10,
    max_coverage=500,
    engine="polars",  # Default: fast, in-memory
)

# 2. Wrap and analyze
mdata = MethylData(adata)
mdata = mdata.filter_coverage(5, 500)
mdata_cpg = mdata.subset_context("CpG")
```

### **Option 2: DuckDB Engine with Sparse Matrices (Memory-Efficient)**
Best for large cohorts (6+ samples, full genome).

```python
# Memory peak: ~4–6 GB instead of ~17–20 GB
adata = epykit.io.build_anndata_streaming(
    sample_ids=["ctrl_1", "ctrl_2", "treat_1", "treat_2"],
    file_paths=["ctrl_1.cov.gz", "ctrl_2.cov.gz", "treat_1.cov.gz", "treat_2.cov.gz"],
    min_coverage=10,
    join_type="outer",
    duckdb_memory_limit="8GB",
    sparse=True,  # <-- 50-70% RAM savings on sparse datasets
)
```

### **Full Analysis Pipeline**

```python
from epykit.core import MethylData
from epykit.stats import calculate_diff_meth, merge_dmrs
from epykit.intervals import tile_counts, annotate_cpg_islands
from epykit.plot import pca, sample_correlation

# 1. Quality control
mdata = MethylData(adata)
mdata = mdata.filter_coverage(min_cov=5, max_cov=500)
mdata_cpg = mdata.subset_context("CpG")

# 2. Exploratory analysis
pca(mdata_cpg.adata, show=True)
sample_correlation(mdata_cpg.adata)

# 3. Differential methylation testing (DMC)
results = calculate_diff_meth(
    mdata_cpg,
    treatment_col="group",
    treatment_val="treated",
    control_val="control",
    method="glm",  # Logistic GLM with LRT
    fdr_threshold=0.05,
)
print(f"Significant sites: {(results['padj'] < 0.05).sum()}")

# 4. Merge into regions (DMR)
sig_sites = results[results["padj"] < 0.05]
dmrs = merge_dmrs(sig_sites, min_cpg=2, max_gap=300)
print(f"Identified {len(dmrs)} DMRs")

# 5. Bin into tiling windows
tiled = tile_counts(mdata_cpg.adata, window=1000, step=1000)
print(f"Created {tiled.n_vars} 1-kb windows")
```

## Testing

Run the test suite to validate the pipeline:

```bash
# Quick test with region restriction (86 sites, ~2 min)
python test_6samples.py

# Full-genome test (42M+ sites, ~30 min)
python test_6samples_full_genome.py

# Complete pipeline with sparse matrices
python test_sparse_full_pipeline.py

# Compare dense vs sparse memory usage
python test_sparse_comparison.py


## Installation

### Using pip (development)
```bash
git clone https://github.com/d-ert/epykit.git
cd epykit
pip install -e ".[dev]"
```

### Using uv (recommended for speed)
```bash
uv pip install -e ".[dev]"
```

### Using conda / Bioconda (when released)
```bash
conda install -c bioconda epykit
```

## Memory Optimization (P1 Features)

EpyKit implements aggressive memory optimizations for large-scale WGBS:

| Feature | Impact | Status |
|---------|--------|--------|
| **Sparse matrix storage** | 50–70% storage reduction for outer-join datasets | ✅ Implemented (`sparse=True` parameter) |
| **DuckDB streaming** | Processes full genomes in constant memory | ✅ Implemented |
| **Per-sample processing** | Eliminates simultaneous multi-sample frames | ✅ Implemented |
| **Pre-allocated arrays** | Eliminates memory spike during concatenation | ✅ Implemented |
| **Compact indexing** | Avoids string materialization overhead | ✅ Implemented |

**Result:** For 6 samples × 42M sites:
- **Before:** ~17–20 GB peak RAM
- **After:** ~4–6 GB peak RAM (70% reduction!)
- **With sparse=True:** Additional 50% savings on output matrix

## Current Implementation Status

### ✅ Fully Implemented
- **I/O:** Bismark, bedGraph, CX_report, generic formats
- **Engines:** Polars (fast), DuckDB (memory-efficient)
- **Core:** MethylData wrapper, coverage filtering, context subsetting, unite operations
- **Statistics:** Fisher exact test, Logistic GLM + LRT, HC0 overdispersion correction, BH-FDR, DMR merging
- **Intervals:** Tiling (fixed-width windows), site-to-tile mapping
- **Plotting:** PCA, UMAP, sample correlation heatmaps
- **P1 optimizations:** All 5 items (sparse matrices, streaming, pre-allocation)

### 🟡 Partially Implemented / In Progress
- **Feature annotation:** Basic `annotate_cpg_islands()` exists but needs testing
- **Zarr persistence:** Underlying support exists, integration being refined
- **Multi-factor design:** Currently GLM tests single treatment column

### 🔴 Planned (P2–P5)
- Chromosome-by-chromosome processing (P2)
- Direct-to-Zarr streaming (P2)
- Beta-binomial GLM (P3)
- Hidden Markov Model for HMRs (P3)
- Tabix-backed flat files (P3)
- Weighted methylation levels (P3)

---

## Project Structure

```
src/epykit/
├── __init__.py              # Top-level package exports
│
├── io/                      # Data ingestion & building
│   ├── __init__.py
│   ├── bismark.py           # read_bismark_coverage(), read_bismark_cx_report()
│   ├── generic.py           # read_bedgraph(), read_generic_methylation()
│   ├── regions.py           # BED file parsing & merging
│   ├── sample_sheet.py      # read_samples() multi-sample loader
│   ├── anndata_builder.py   # Polars-based AnnData builder (fast, in-memory)
│   └── anndata_builder_duckdb.py  # DuckDB streaming builder (memory-efficient)
│
├── core/                    # Analysis core & data structure wrapper
│   ├── __init__.py
│   └── methyldata.py        # MethylData class, filter_coverage(), subset_context(), unite()
│
├── intervals/               # Genomic interval operations
│   ├── __init__.py
│   └── tiling.py            # tile_counts(), annotate_features(), annotate_cpg_islands()
│
├── stats/                   # Statistical testing
│   ├── __init__.py
│   ├── tests.py             # fisher_exact_test(), glm_lrt_test(), calculate_diff_meth()
│   └── dmr.py               # merge_dmrs() for DMR identification
│
└── plot/                    # Visualization & QC
    ├── __init__.py
    └── qc.py                # pca(), umap(), sample_correlation()
```

## Architecture & Data Model

EpyKit uses the **AnnData** specification with 0-based, half-open (BED-style) coordinates:

```python
adata.X                     # Beta-value matrix (n_samples × n_sites) | float32 | range [0–1]
adata.obs                   # Sample metadata (group, batch, age, etc.) | DataFrame
adata.var                   # Site coordinates (chr, start, end, strand, context) | DataFrame
adata.layers['coverage']    # Total read coverage matrix | int32
adata.layers['methylated_counts']  # Methylated read counts | int32
```

**Sparse matrices:** When `sparse=True`, `X` is stored as `scipy.sparse.csr_matrix`, reducing memory 50–70% for outer-join datasets.

## Comparison: EpyKit vs methylKit

| Aspect | methylKit (R) | EpyKit (Python) |
|--------|--------------|-----------------|
| **Data backend** | tabix flat files | AnnData (in-RAM or backed) |
| **Processing engine** | data.table | Polars (Rust, 10–100× faster) |
| **Interval ops** | GenomicRanges | polars-bio / PyRanges |
| **Memory model** | Chunk-based | DuckDB streaming or sparse matrices |
| **Peak RAM (6 samples × 42M sites)** | ~17–20 GB | ~4–6 GB (70% savings!) |
| **Overdispersion** | quasi-binomial | HC0 robust SE (equivalent) |
| **ML integration** | ❌ | ✅ scverse / scikit-learn compatible |
| **Cloud storage** | 🟡 Via external tools | ✅ Zarr native |

## Example: End-to-End Differential Methylation Analysis

```python
import pandas as pd
from pathlib import Path
from epykit.io import build_anndata_streaming
from epykit.core import MethylData
from epykit.stats import calculate_diff_meth, merge_dmrs
from epykit.plot import pca, sample_correlation

# Load sample sheet
ss = pd.read_csv("sample_sheet.csv")
sample_ids = list(ss["sample_id"])
file_paths = [Path(row["path"]) for _, row in ss.iterrows()]
obs_metadata = ss[["group"]].set_index(pd.Index(ss["sample_id"]))

# Build AnnData with sparse matrices & DuckDB streaming
adata = build_anndata_streaming(
    sample_ids=sample_ids,
    file_paths=file_paths,
    obs_metadata=obs_metadata,
    min_coverage=10,
    max_coverage=500,
    join_type="outer",
    duckdb_memory_limit="8GB",
    sparse=True,  # <- 50-70% RAM savings
)

# Quality control
mdata = MethylData(adata)
pca(mdata.adata, show=True)
sample_correlation(mdata.adata)

# Filter & subset
mdata = mdata.filter_coverage(5, 500)
mdata_cpg = mdata.subset_context("CpG")

# Differential methylation testing (DMC)
results = calculate_diff_meth(
    mdata_cpg,
    treatment_col="group",
    treatment_val="treated",
    control_val="control",
    method="glm",
    fdr_threshold=0.05,
)

# Identify regions (DMR)
sig = results[results["padj"] < 0.05]
dmrs = merge_dmrs(sig, min_cpg=2, max_gap=300)

print(f"✓ {len(results)} sites tested")
print(f"  → {len(sig)} significant (padj < 0.05)")
print(f"  → {len(dmrs)} DMR regions")
```

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

### Development Setup

```bash
# Clone and enter the repo
git clone https://github.com/d-ert/epykit.git
cd epykit

# Create virtual environment & install with dev dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run full pipeline test
python test_sparse_full_pipeline.py
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Questions?** Open an issue on [GitHub](https://github.com/d-ert/epykit/issues).
