# EpyKit 🧬

A highly scalable, production-grade Python framework for **Whole Genome Bisulfite Sequencing (WGBS)** data analysis. Built as a modern Python alternative to R's `methylKit`, offering superior performance through Rust-powered data processing and cloud-native storage.

## Features

- 🚀 **High-performance ingestion** — Polars lazy CSV reader with predicate pushdown for Bismark, bedGraph, bwa-meth, and generic formats
- 🗄️ **AnnData-native data model** — Zarr/HDF5-backed lazy loading for cohort-scale data without RAM exhaustion
- 🔬 **Ergonomic API** — `MethylData` wrapper with typed accessors for beta values, coverage, and site metadata
- 📐 **Interval algebra** — `polars-bio` powered tiling windows, feature annotation, CpG island classification
- 📊 **Robust statistics** — Fisher exact, Logistic GLM + LRT, HC0 overdispersion correction, BH-FDR, DMR merging
- 🔍 **QC & EDA** — PCA with batch detection, UMAP, sample correlation heatmaps
- 📦 **Modern packaging** — `uv` + `hatchling` + `hatch-vcs`, deployable via PyPI and Bioconda

## Supported Aligners / Formats

| Aligner | Format | Reader function |
|---------|--------|----------------|
| Bismark | `.bismark.cov.gz` / `.cov` | `epykit.io.read_bismark_coverage()` |
| bwa-meth | bedGraph | `epykit.io.read_bedgraph()` |
| Bismark | `CX_report` | `epykit.io.read_bismark_cx_report()` |
| Generic | Tab-separated | `epykit.io.read_generic_methylation()` |

## Quick Start

### For small to medium cohorts

```python
import epykit

# 1. Load a single Bismark coverage file
df = epykit.io.read_bismark_coverage("sample1.bismark.cov.gz", min_coverage=5)

# 2. Load multiple samples from a sample sheet (standard Polars engine)
adata = epykit.io.read_samples("sample_sheet.csv", min_coverage=5)
```

### For large cohorts (memory-efficient DuckDB engine)

For 42M+ loci and 6+ samples, use DuckDB streaming to minimize peak RAM:

```python
# Memory-efficient: ~4.5–6 GB peak RAM (vs. 17–20 GB for default)
adata = epykit.io.read_samples(
    "sample_sheet.csv",
    min_coverage=5,
    engine="duckdb"
)

# Optional: save to persistent Zarr storage
adata = epykit.io.read_samples(
    "sample_sheet.csv",
    min_coverage=5,
    engine="duckdb",
    output="zarr",
    out_path="cohort.zarr"
)
```

### Continue analysis

```python
import epykit
adata = epykit.io.read_samples("sample_sheet.csv", min_coverage=5)

# 3. Wrap in MethylData for ergonomic analysis
mdata = epykit.core.MethylData(adata)

# 4. Filter and subset
mdata = mdata.filter_coverage(min_cov=5, max_cov=500)
mdata_cpg = mdata.subset_context("CpG")
mdata_united = mdata_cpg.unite(type="intersect")

# 5. Tiling windows
tiled = epykit.intervals.tile_counts(mdata_united.adata, window=1000, step=1000)

# 6. Differential methylation
results = epykit.stats.calculate_diff_meth(
    mdata_united,
    treatment_col="group",
    test="auto",      # auto-selects Fisher or GLM based on replicates
    overdispersion=True,
    fdr_method="BH",
)

# 7. QC plots
epykit.plot.pca(mdata_united.adata)
epykit.plot.sample_correlation(mdata_united.adata)
```

## Installation

### Using pip (development)
```bash
pip install -e ".[dev]"
```

### Using uv (recommended)
```bash
uv pip install -e ".[dev]"
```

### Using conda / Bioconda
```bash
conda install -c bioconda -c conda-forge EpyKit
```

## Project Structure

```
src/epykit/
├── __init__.py          # Top-level exports
├── io/                  # Data ingestion (Bismark, bedGraph, sample sheets)
│   ├── __init__.py
│   ├── bismark.py       # read_bismark_coverage(), read_bismark_cx_report()
│   ├── generic.py       # read_bedgraph(), read_generic_methylation()
│   └── sample_sheet.py  # read_samples() multi-sample loader
├── core/                # MethylData wrapper and core operations
│   ├── __init__.py
│   └── methyldata.py    # MethylData class, filter_coverage, unite, subset_context
├── intervals/           # Genomic interval algebra (polars-bio)
│   ├── __init__.py
│   └── tiling.py        # tile_counts, annotate_features, cpg_islands
├── stats/               # Statistical testing engine
│   ├── __init__.py
│   ├── tests.py         # Fisher exact, GLM+LRT, HC0
│   └── dmr.py           # Multiple testing correction, DMR merging
└── plot/                # Visualization
    ├── __init__.py
    └── qc.py            # PCA, UMAP, heatmaps, coverage plots
```

## Architectural Design

This package is built on the **AnnData** specification:

- **`adata.X`** — Beta-value matrix (methylation %) | shape: `(n_samples, n_sites)`
- **`adata.obs`** — Sample metadata (group, batch, age, etc.)
- **`adata.var`** — Site coordinates (chr, start, end, strand, context)
- **`adata.layers['coverage']`** — Total read coverage matrix
- **`adata.layers['methylated_counts']`** — Raw methylated read count matrix

## Comparison with methylKit

| Feature | methylKit (R) | EpyKit (Python) |
|---------|--------------|---------------------------|
| Data backend | tabix flat files | AnnData + Zarr (lazy, cloud-native) |
| Tabular engine | data.table | Polars (Rust, 10-100× faster) |
| Interval ops | GenomicRanges | polars-bio (282× faster on multicore) |
| Overdispersion | quasi-binomial | HC0 robust SE (mathematically equivalent) |
| Out-of-core | chunk.size param | Zarr chunked lazy loading |
| ML integration | ❌ | ✅ scverse / scikit-learn compatible |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License — see [LICENSE].
