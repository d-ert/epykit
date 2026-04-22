# py-methyl-toolkit 🧬

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
| Bismark | `.bismark.cov.gz` / `.cov` | `pymethyl.io.read_bismark_coverage()` |
| bwa-meth | bedGraph | `pymethyl.io.read_bedgraph()` *(coming soon)* |
| Bismark | `CX_report` | `pymethyl.io.read_bismark_cx_report()` *(coming soon)* |
| Generic | Tab-separated | `pymethyl.io.read_generic_methylation()` |

## Quick Start

```python
import pymethyl

# 1. Load a single Bismark coverage file
df = pymethyl.io.read_bismark_coverage("sample1.bismark.cov.gz", min_coverage=5)

# 2. Load multiple samples from a sample sheet
adata = pymethyl.io.read_samples("sample_sheet.csv", min_coverage=5)

# 3. Wrap in MethylData for ergonomic analysis
mdata = pymethyl.core.MethylData(adata)

# 4. Filter and subset
mdata = mdata.filter_coverage(min_cov=5, max_cov=500)
mdata_cpg = mdata.subset_context("CpG")
mdata_united = mdata_cpg.unite(type="intersect")

# 5. Tiling windows
tiled = pymethyl.intervals.tile_counts(mdata_united.adata, window=1000, step=1000)

# 6. Differential methylation
results = pymethyl.stats.calculate_diff_meth(
    mdata_united,
    treatment_col="group",
    test="auto",      # auto-selects Fisher or GLM based on replicates
    overdispersion=True,
    fdr_method="BH",
)

# 7. QC plots
pymethyl.plot.pca(mdata_united.adata)
pymethyl.plot.sample_correlation(mdata_united.adata)
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
conda install -c bioconda -c conda-forge py-methyl-toolkit
```

## Project Structure

```
src/pymethyl/
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

| Feature | methylKit (R) | py-methyl-toolkit (Python) |
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

MIT License — see [LICENSE](LICENSE).
