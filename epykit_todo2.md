


Here is the updated, highly detailed roadmap incorporating the exact timelines, code-level implementation details, and RAM impacts. This gives you a clear, week-by-week sprint plan to fully unlock the `scverse` ecosystem.

---

# EpyKit — The AnnData-Native WGBS Framework & Development Roadmap

> **Architectural Vision:** EpyKit is built on `AnnData`, effectively docking bulk WGBS (Whole Genome Bisulfite Sequencing) into the most powerful, active ecosystem in modern Python bioinformatics (the `scverse`). 
>
> Traditionally, bulk methylation packages use rigid, custom S4 classes (e.g., in R). By leveraging `AnnData`, EpyKit seamlessly handles complex metadata bookkeeping, multi-layer storage (beta, coverage, counts), natively supports sparse matrices/disk-backed data, and gives users "free" dimensionality reduction via `Scanpy` and multi-omics capabilities via `MuData`.

---

## 🏆 Recent Milestones Reached
✅ **P0 — Bugs:** `filter_coverage` logic fixed, `tile_counts` vectorized, Polars bio-overlaps fixed, Zarr silent data loss patched.
✅ **P1 — RAM Immediate Wins:** Avoided double-materializing beta matrices, fixed string index RAM bloat, fixed chunk accumulation memory spikes, implemented streaming union for Polars builder.

---

## 🚀 The "AnnData Universe" Integration Timeline

While EpyKit correctly utilizes AnnData for multi-layer storage and metadata synchronization, we are currently leaving its most powerful features—true sparse computation, out-of-core (backed) processing, and ecosystem integrations—underutilized. 

### ⏱️ Week 1–2: True Sparse Matrix Compute (Property Layer)
*Current State: `sparse=True` works for storage, but accessors call `.toarray()`, instantly blowing up RAM.*

* **Sparse-Transparent Property Layer in `MethylData`:**
  * Rewrite `.beta`, `.coverage`, and `.methylated` properties. They unconditionally call `.toarray()` today—they must be rewritten to return sparse arrays when `adata.X` is sparse, and only densify on explicit request.
  * **Fix Implicit Densification:** `filter_coverage`, `unite`, and `subset_context` internally call `self.coverage` before slicing, which densifies the whole matrix. Rewrite these to index `adata.layers['coverage']` directly without materializing it.
  * **Add `.to_dense()`:** Give callers explicit control over when densification happens. This replaces the current implicit and dangerous behavior scattered across the three property accessors.

### ⏱️ Week 2–4: Sparse-Aware Stats Engine
*Current State: GLM and Fisher tests pull the entire dataset into memory.*

* **Refactor Stats Engine:** `glm_lrt_test` and `fisher_exact_test` currently pull full dense matrices via `mdata.methylated`. Rewrite these to iterate site batches using `adata.layers['coverage'][:, i]`, which works natively on both sparse and dense arrays without an upfront `.toarray()`.
* **RAM Impact:** Working memory during testing drops from `O(n_samples × n_sites)` held in RAM to `O(n_samples × batch_size)`. For 42 M sites, that is the difference between **~7 GB and ~200 MB** of working memory during the GLM loop.

### ⏱️ Month 2: Out-of-Core / Backed Mode for Large Cohorts
*Current State: EpyKit can save/load `.h5ad`, but downstream functions use `np.asarray()` which defeats backing.*

* **Wire `backed='r'` Properly:** Pass the backed flag through `epykit.io.load()` correctly. The filter and stats pipelines currently call `np.asarray(adata.X)` which loads the whole matrix into RAM. Replace these with chunked slice reads from the HDF5/Zarr file.
* **Add RAM Footprint Table to Docs:** Document the minimum dense footprint: `n_samples × n_sites × 12 bytes` (beta + coverage + methylated at float32/int32). For a cohort of 200 samples × 25 M sites = **60 GB**. This explicitly shows users why backed mode becomes mandatory at that scale.
* **Direct-to-Zarr Streaming (Zero-copy):** Open Zarr arrays at the start of dataset construction. Run the DuckDB JOIN and write directly to `zarr[:, locus_idx]`.
* **Chromosome-by-Chromosome Processing:** Process massive datasets one chromosome at a time. Write chromosome slices to a Zarr store and concatenate at the end via `anndata.experimental.concat`.

### ⏱️ Month 2–3: Scanpy-First Plotting (Remove the Wrappers)
*Current State: Custom boilerplate masking Scanpy's native power.*

* **Remove Custom Plotting Logic:** Replace `epykit.plot.qc` with thin delegates. For example, `epykit.plot.pca(mdata)` should become a one-liner forwarding to `sc.pp.pca` and then `sc.pl.pca`. This **removes ~300 lines** of NaN imputation loops and manual scatter code you are currently maintaining yourself for free.
* **Document the `scverse` Call Chain:** Add a section to the README showing users how to directly interface with scanpy:
  * `sc.pp.pca(mdata.adata)`
  * `sc.tl.umap(mdata.adata)`
  * `sc.pl.umap(mdata.adata, color='group')`
  * Users get immediate access to publication-ready plots without EpyKit needing to own any of the plotting logic.

### ⚡ Quick Win: MuData Multi-Omics Bridge
*Current State: No mention of `MuData`. Massive missed opportunity.*

* **Docs Only, Zero Code Changes Needed:** Add a README section showing how easily EpyKit integrates with multi-omics using `muon`:
  ```python
  from mudata import MuData
  mdata = MuData({'meth': meth_data.adata, 'rna': rna_adata})
  ```
  This is the entire integration. Because EpyKit already outputs standard AnnData, it just works.
* **Embeddings Best Practices:** Document that EpyKit stores PCA/UMAP embeddings consistently in `adata.obsm['X_pca']` (which you already do). This means they are immediately available to `muon`'s Multi-Omics Factor Analysis (MOFA) and joint dimensionality reduction.

---

## 🔬 Core WGBS Features & Statistical Roadmap (Medium-Term Backlog)

Beyond ecosystem integration, the core biological and statistical algorithms require the following expansions:

### 1. Statistical Methods
* **Beta-binomial GLM:** Implement a proper beta-binomial regression model (like `dnmtools radmeth`) that directly models overdispersion without ad-hoc HC0 correction.
* **HMM for Hypomethylated Regions (HMR):** Implement a two-state HMM (methylated / unmethylated) to identify contiguous hypomethylated regions (the most principled DMR approach for WGBS).
* **Partially Methylated Domains (PMDs):** Add `epykit.stats.find_pmds()` to identify 100kb+ domains of intermediate methylation.
* **Multi-factor Design Matrices:** Extend `calculate_diff_meth()` to accept R-style formula strings (e.g., `"~ group + batch + age"`) using `patsy`.

### 2. I/O & Formatting
* **bismark2bedGraph Lazy Reading:** Add a `lazy=True` mode for `read_bismark_coverage` returning a `pl.LazyFrame` to avoid full materialization for subset queries.
* **Native MethylDackel / bwa-meth Support:** Add a dedicated reader for MethylDackel's `--mergeContext` output.
* **Pre-convert to Indexed Parquet:** Create `epykit.io.convert_to_parquet()` to convert heavy CSV files into Parquet, unlocking DuckDB's predicate pushdown on `chr` and `start`.

---

## 📋 Priority Summary

| Timeline | Category | Goal | Code Impact | RAM Impact |
|----------|----------|------|-------------|------------|
| **Completed** | 🔴 Bug Fixes | Core fixes, Polars unions, vectorized loops | High | Medium |
| **Week 1-2** | 🟡 AnnData | **Sparse Property Layer** (`.to_dense()`, direct layer indexing) | Medium | High |
| **Week 2-4** | 🟡 AnnData | **Sparse Stats Engine** (Batch iterations on layers) | High | **7GB → 200MB** |
| **Month 2** | 🟡 AnnData | **Backed Mode & Zarr** (Fix `np.asarray`, docs on 60GB footprint) | Medium | Out-of-core |
| **Month 2-3** | 🟢 Ecosystem | **Scanpy Plotting** (Delete 300 lines of custom plotting) | Negative (Deletion) | N/A |
| **Quick Win** | 🟢 Ecosystem | **MuData Docs** (`MuData({'meth': ...})` vignette) | Docs Only | N/A |
| **Backlog** | 🔵 Bio/Stats | HMM, Beta-binomial GLM, Parquet IO | High | Low |