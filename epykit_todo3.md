# EpyKit — AnnData-Native WGBS Analysis

## Overview

EpyKit is a Python toolkit for bulk whole-genome bisulfite sequencing (WGBS) analysis built **directly on the AnnData/scverse ecosystem**. Rather than reinventing a custom container, EpyKit uses `AnnData` as its core data model, giving users immediate access to mature tooling for metadata management, dimensionality reduction, storage backends, and multi-omics integration.

This means EpyKit is not just a methylation package—it is a **WGBS engine that plugs into the broader Python bioinformatics universe**.

---

## Why AnnData Was the Right Architecture Choice

### Automatic Metadata Synchronization

Every filtering or subsetting operation preserves alignment across:

* `adata.X` — beta values / methylation fractions
* `adata.obs` — sample metadata
* `adata.var` — genomic locus metadata
* `adata.layers` — coverage and methylated counts

When users subset samples or loci, all matrices and annotations remain synchronized automatically.

### Multi-Layer Native Storage

EpyKit stores multiple assay representations in one object:

* `X` = beta values
* `layers['coverage']`
* `layers['methylated_counts']`

This enables seamless movement between statistical testing, machine learning, and visualization.

### Immediate Access to Scanpy / scverse

Because EpyKit outputs standard AnnData objects, users can directly run:

```python
sc.pp.pca(mdata.adata)
sc.tl.umap(mdata.adata)
sc.pl.umap(mdata.adata, color='group')
```

No conversion step required.

### Large Dataset Storage (.h5ad / Zarr)

AnnData provides compressed persistent formats:

* `.h5ad` for portable single-file storage
* `Zarr` for chunked cloud/HPC workflows
* backed mode for datasets larger than RAM

### Multi-Omics Ready (MuData)

EpyKit methylation objects can be combined with RNA-seq or ATAC-seq instantly:

```python
from mudata import MuData
mdata = MuData({'meth': meth_adata, 'rna': rna_adata})
```

No custom integration layer needed.

---

# Current Status

## Completed

### P0 — Core Correctness Fixes

* Metadata synchronization architecture complete
* Layered storage complete
* Core AnnData container stable
* Filtering returns safe copied views

### P1 — Immediate Infrastructure Wins Complete

* Major correctness/engineering cleanup complete
* Foundation ready for sparse-aware refactor

---

# AnnData-First Roadmap

## Week 1–2: Sparse-Transparent MethylData Layer

### Goal

Make sparse matrices a real feature throughout the API.

### Current Limitation

Properties such as:

* `.beta`
* `.coverage`
* `.methylated`

currently densify sparse matrices implicitly.

### Planned Changes

* Return sparse arrays when underlying storage is sparse
* Only densify when explicitly requested
* Add:

```python
mdata.to_dense()
```

* Rewrite filtering methods (`filter_coverage`, `subset_context`, `unite`) to slice layers directly instead of materializing full dense arrays.

### Result

Sparse mode becomes useful for both storage **and computation**.

---

## Week 2–4: Sparse-Aware Statistical Engine

### Goal

Allow differential methylation testing without loading whole matrices into RAM.

### Planned Changes

Refactor:

* `glm_lrt_test()`
* `fisher_exact_test()`

from dense full-matrix access to batched column-wise operations:

```python
adata.layers['coverage'][:, i:j]
adata.layers['methylated_counts'][:, i:j]
```

### Expected Impact

Working memory reduces from:

`O(samples × all_sites)`

to:

`O(samples × batch_size)`

Example:

* 42M loci dense workflow ≈ several GB RAM
* batched sparse workflow ≈ a few hundred MB

---

## Month 2: True Backed / Out-of-Core Mode

### Goal

Enable cohorts too large for RAM.

### Planned Changes

Expose backed loading properly:

```python
epykit.io.load(path, backed='r')
```

Then refactor pipelines to use chunk reads instead of:

```python
np.asarray(adata.X)
```

### Why It Matters

Approximate dense footprint:

`n_samples × n_sites × 12 bytes`

(beta + coverage + methylated)

Example:

* 200 samples × 25M loci ≈ 60 GB+

Backed mode becomes mandatory at that scale.

---

## Month 2–3: Scanpy-First Plotting

### Goal

Stop maintaining custom plotting wrappers where Scanpy already excels.

### Planned Changes

Replace large internal plotting code with thin delegates:

```python
epykit.plot.pca(mdata)
epykit.plot.umap(mdata)
```

internally calling Scanpy.

### Benefits

* Less maintenance burden
* Better defaults
* Publication-ready figures
* Users stay inside scverse standards

---

## Quick Win: MuData Bridge (Docs Only)

No code changes needed.

Add documented workflows:

```python
MuData({'meth': mdata.adata, 'rna': rna_adata})
```

Use existing embeddings stored in:

* `adata.obsm['X_pca']`
* `adata.obsm['X_umap']`

for downstream MOFA / joint factor models.

---

# Long-Term Vision

EpyKit becomes the **standard AnnData-native methylation toolkit** for Python.

Users should be able to:

```python
# Build methylation matrix
meth = epykit.read_samples(...)

# QC + embedding
sc.pp.pca(meth.adata)
sc.tl.umap(meth.adata)

# Differential methylation
epykit.stats.calculate_diff_meth(...)

# Multi-omics integration
MuData({'meth': meth.adata, 'rna': rna})
```

---

# Why This Matters

Most methylation tools live in isolated ecosystems with custom containers.

EpyKit can instead own a unique niche:

> High-performance WGBS analysis built natively inside scverse.

That means:

* better interoperability
* easier machine learning workflows
* shared visualization tools
* modern storage backends
* future-proof ecosystem growth

---

# Strategic Priority Order

1. Sparse-transparent API layer
2. Sparse-aware stats engine
3. Backed mode pipelines
4. Scanpy-first plotting cleanup
5. MuData examples and tutorials
6. Advanced multi-omics workflows

---

# Bottom Line

Choosing AnnData was not just convenient—it was strategically correct.

EpyKit already has the hard part: WGBS-specific parsing, aggregation, and testing.
The next phase is maximizing the ecosystem advantages that AnnData already gives for free.
