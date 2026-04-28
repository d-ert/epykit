# EpyKit — Comprehensive Improvement TODO

> Informed by analysis of epykit source code + approaches from **methylKit** (tabix-backed DB objects, per-chromosome chunking), **dnmtools** (streaming sorted merge, binary formats), **bsseq/BiocMAP** (HDF5Array backed matrices), and **wgbstools** (custom 100× compressed binary format).

---

## 🔴 P0 — Bugs (wrong results today)

### 1. `filter_coverage` — `require_all_samples` has no effect
**File:** `src/epykit/core/methyldata.py` lines 153–158

Both branches of the `if require_all_samples` block execute the **same** expression:
```python
# BUG: both branches are identical
if require_all_samples:
    mask = np.all(cov >= min_cov, axis=0)
else:
    mask = np.all(cov >= min_cov, axis=0)   # <— should be np.any
```
Fix: the `else` branch should be `mask = np.any(cov >= min_cov, axis=0)` — keep a site if *at least one* sample meets the threshold, which is the lenient interpretation.

---

### 2. `tile_counts` — row-by-row Python loop over potentially millions of (site, tile) pairs
**File:** `src/epykit/intervals/tiling.py` lines 141–148

```python
for _, row in s2t.iterrows():   # O(n) Python loop — catastrophically slow at scale
    s_i = int(row["_site_idx"])
    t_i = tile_idx_map.get(row["_tile_key"], -1)
    ...
    tile_cov[:, t_i] += cov_arr[:, s_i]
    tile_meth[:, t_i] += meth_arr[:, s_i]
```
Fix: vectorise with NumPy fancy indexing — convert both index columns to integer arrays, then use `np.add.at` or pre-build a sparse site→tile mapping matrix.

```python
site_indices = s2t["_site_idx"].to_numpy(dtype=np.int32)
tile_indices  = s2t["_tile_key"].map(tile_idx_map).to_numpy(dtype=np.int32)
valid = tile_indices >= 0
for t in np.unique(tile_indices[valid]):
    mask = (tile_indices == t) & valid
    tile_cov[:, t]  = cov_arr[:, site_indices[mask]].sum(axis=1)
    tile_meth[:, t] = meth_arr[:, site_indices[mask]].sum(axis=1)
```

---

### 3. `annotate_cpg_islands` — always uses pure-Polars fallback for shore/shelf
**File:** `src/epykit/intervals/tiling.py` `annotate_cpg_islands()`

The helper `_get_overlapping_sites` calls `_overlap_pure_polars` unconditionally — even when `_HAS_POLARS_BIO = True`. This silently falls back to the slow cross-join path for every island classification call. Fix by routing through `_overlap_polars_bio` when available.

---

### 4. `read_samples(output='zarr')` — silent data loss risk
**File:** `src/epykit/io/sample_sheet.py`

The `index_name` equality check uses `.equals()` on a pandas Series vs an Index object — these types are not directly comparable and the check may always evaluate `False`, silently skipping the rename that prevents Zarr write collisions.

---

## 🟠 P1 — RAM: Immediate Wins (no architecture change)

### 5. Avoid double-materialising the beta matrix in `build_anndata_streaming`
**File:** `src/epykit/io/anndata_builder_duckdb.py`

`beta_mat` is allocated as `float32` (good), but `AnnData(X=beta_mat, ...)` copies the array internally during construction. Pass `X=beta_mat` **and** immediately `del beta_mat` before the AnnData constructor runs, or use `ad.AnnData.__init__` with `dtype` hints. Better yet, write directly to a Zarr store during Step 3 (see P2 item 11).

### 6. `var_df` string index wastes ~2 GB for 42 M loci
**File:** `src/epykit/io/anndata_builder_duckdb.py`

The `var_df` is assigned `pd.RangeIndex` (integers) but then `AnnData` converts `var_names` to strings. For 42 M sites that is ~42 M × ~15 bytes ≈ 600 MB just for the index. Fix: keep `var_names` as integer strings only during I/O; use a `locus_id` int64 column as the true key and set `var_df.index` to a `pd.RangeIndex` — AnnData will store it compactly.

### 7. `chr_codes_arr` chunk accumulation in Step 4 holds two copies
**File:** `src/epykit/io/anndata_builder_duckdb.py` Step 4b

The loop accumulates `chr_codes_list` and `start_list`, then calls `np.concatenate`. At the moment of concatenation, both the list-of-chunks and the result array exist simultaneously (2× peak memory). Fix: pre-allocate the output arrays before the loop and fill by slice:
```python
chr_codes_arr = np.empty(n_sites, dtype=np.int8)
start_arr     = np.empty(n_sites, dtype=np.int32)
offset = 0
for chunk in ...:
    n = len(chunk)
    chr_codes_arr[offset:offset+n] = chunk["chr_code"]
    start_arr[offset:offset+n]     = chunk["start"]
    offset += n
```

### 8. `build_anndata` (Polars engine) loads all sample DataFrames simultaneously
**File:** `src/epykit/io/anndata_builder.py`

All N sample DataFrames (passed in as a list) live in RAM simultaneously before `keyed_frames` is built. For 6 × 30 M rows that is ~14 GB. Fix: process samples one-at-a-time — compute the locus union from a streaming pass that never holds more than two frames at once (the same streaming-UNION pattern already implemented in the DuckDB engine).

### 9. Sparse storage for beta matrix when coverage is sparse
Both builders always create dense `float32` beta matrices. For outer-join datasets, many cells are NaN (no coverage). A `scipy.sparse.csr_matrix` or `anndata`'s native sparse support can cut this to 20–50% of dense size for typical WGBS outer joins. Add a `sparse=True` flag that flows all the way through to `AnnData(X=csr_matrix(...))`.

---

## 🟡 P2 — Architecture: Medium-Term RAM Solutions

### 10. Write directly to Zarr during DuckDB streaming (zero-copy assembly)
Inspired by **methylKit's tabix-backed DB objects** and **bsseq's HDF5Array backend**.

Instead of pre-allocating `beta_mat` / `cov_mat` / `meth_mat` in RAM and then writing, open the Zarr arrays at the start and fill them slice-by-slice per sample:
```
Step 2: zarr.open_group(path) → create beta/cov/meth datasets (n_samples × n_sites)
Step 3: for each sample → DuckDB JOIN → scatter result → write zarr[:, locus_idx]
Step 4: ad.read_zarr(path, backed=True)   # zero extra RAM
```
Peak RAM drops from `3 × n_samples × n_sites × 4 bytes` to `~1 × n_sites × 4 bytes` (one sample's worth).

### 11. Chromosome-by-chromosome processing (dnmtools/methylKit pattern)
**dnmtools `merge`** and **methylKit's `applyTbxByChr`** both process one chromosome at a time, keeping peak RAM proportional to the largest chromosome (chr1 ≈ 10% of genome).

Implement a `build_anndata_by_chr()` variant:
- Iterate over chromosomes
- For each chromosome, run the DuckDB UNION/JOIN only on that subset
- Write chromosome slices to a Zarr store
- Concatenate at the end via `anndata.experimental.concat`

This reduces peak RAM by ~10× compared to whole-genome assembly.

### 12. Pre-convert Bismark files to indexed Parquet/Arrow IPC
**dnmtools** uses a custom binary format achieving 80–95% storage reduction. **methylKit** uses bgzip+tabix.

For epykit, Parquet is the practical choice:
- One-time conversion: `bismark.cov.gz` → `sample.parquet` (partitioned by chromosome)
- DuckDB reads Parquet natively with predicate pushdown on `chr` and `start`
- No regex-heavy CSV parsing per analysis run
- Enable `regions_bed` filtering at the Parquet scan level (partition pruning)

Add `epykit.io.convert_to_parquet(path_in, path_out)` and auto-detect `.parquet` in `read_samples`.

### 13. Backed AnnData mode throughout the API
AnnData supports `backed="r"` mode for HDF5, which memory-maps the matrix arrays. Ensure that:
- `MethylData` works correctly when `adata.isbacked` is `True`
- `filter_coverage`, `subset_context`, `unite` yield new backed AnnDatas (not in-memory copies)
- Stats functions access `adata.X` in chunks, not via `np.asarray(adata.X)` (which forces full load)

### 14. Chunked GLM testing
**File:** `src/epykit/stats/tests.py`

`glm_lrt_test` iterates over `n_sites` in a Python for-loop. For 42 M sites this is hours of compute and holds all matrices in RAM. Fix:
- Process sites in chunks of ~100k
- Release each chunk's intermediate statsmodels objects after the p-value is extracted
- Add a `chunk_size` parameter and progress bar (tqdm)
- Long term: vectorise the GLM using a direct score-test formulation (eliminates per-site model fitting)

---

## 🟢 P3 — New Features & Statistical Methods

### 15. Beta-binomial GLM (radmeth approach)
**dnmtools `radmeth`** uses a beta-binomial regression model that directly models overdispersion without ad-hoc HC0 correction. The beta-binomial is the proper likelihood for methylation data (reads are not independent due to shared genomic context).

Add `epykit.stats.beta_binomial_test()` using `statsmodels`'s `BetaBinomialP` or the `betareg` formulation.

### 16. Hidden Markov Model for Hypomethylated Regions (HMR)
**dnmtools `hmr`** uses a two-state HMM (methylated / unmethylated) to identify contiguous hypomethylated regions — the most principled DMR approach for WGBS.

Add `epykit.stats.hmr()` wrapping `hmmlearn` or a Cython HMM.

### 17. Tabix-backed flat file format (methylKit parity)
Implement `epykit.io.MethylDB` — a lightweight wrapper that stores per-sample data as bgzip+tabix files (via `pysam`) and exposes a `fetch(chr, start, end)` API. This enables:
- Sub-region queries without loading the whole genome
- Streaming unite() by chromosome
- Interoperability with genome browsers

### 18. Weighted methylation levels (dnmtools `levels` parity)
**dnmtools** distinguishes three methylation level definitions:
- **Weighted** (total methylated reads / total coverage across all sites)
- **Unweighted** (mean of per-site fractions, i.e. mean beta)
- **Fractional** (number of fully methylated sites / total sites)

Currently `global_methylation()` only computes unweighted. Add all three to `MethylData.global_methylation()`.

### 19. Partially Methylated Domains (PMDs)
**dnmtools `pmd`** identifies large-scale (100 kb+) domains of intermediate methylation, a hallmark of cancer and aging. Add `epykit.stats.find_pmds()`.

### 20. Multi-factor design matrix support in GLM
Currently the GLM only supports a single binary `treatment_col`. Extend `calculate_diff_meth()` to accept an R-style formula string (`"~ group + batch + age"`) using `patsy` for design matrix construction.

---

## 🔵 P4 — I/O Improvements

### 21. bismark2bedGraph streaming reader
The current `read_bismark_coverage` loads eagerly via `.collect()`. For very large files, add a `lazy=True` mode that returns a `pl.LazyFrame` — callers that only need a region or a subsample can avoid materialising the full file.

### 22. Support for bwa-meth / MethylDackel output natively
Add a dedicated `read_methyldackel()` reader — MethylDackel's `--mergeContext` output is increasingly common and differs slightly from the generic reader in column semantics.

### 23. Multi-threaded Bismark parsing
The current Polars-engine `read_samples` uses `ThreadPoolExecutor` but Polars CSV scanning already releases the GIL. Profile whether increasing `n_workers` beyond `cpu_count // 2` actually helps or causes memory contention.

### 24. `regions_bed` support in Polars-engine `read_samples`
Currently `regions_bed` is only propagated into DuckDB engine reads. The Polars engine passes `regions_df` into `reader_kwargs` but `read_bismark_cx_report` does accept it — verify the plumbing is correct for all four reader paths and add an integration test.

### 25. Zarr v3 compatibility
`anndata_builder_chunked.py` calls `zarr.open_group` which behaves differently between Zarr v2 and v3. Pin to a compatibility shim or explicitly pass `zarr_version=2`.

---

## ⚙️ P5 — Code Quality & Testing

### 26. Unit test for `filter_coverage` correctness (would have caught bug #1)
Add a pytest fixture with a 3-sample × 5-site AnnData where the correct `require_all_samples=True` and `require_all_samples=False` results differ, and assert both outcomes.

### 27. Memory regression tests
Add a pytest-memray or `tracemalloc`-based test that asserts peak RAM for building a small (3 samples × 1 M sites) AnnData stays below a configurable threshold (e.g. 500 MB).

### 28. Benchmark suite
Add a `benchmarks/` directory with `asv` (airspeed velocity) benchmarks for:
- `read_bismark_coverage` (1 sample, 30 M sites)
- `build_anndata` (6 samples, outer join)
- `tile_counts` (1000-bp windows)
- `calculate_diff_meth` (10 k sites, GLM)

### 29. Property-based tests for interval algebra
`tile_counts` and `annotate_features` have complex edge cases (empty chromosomes, single-site windows, sites at chromosome boundaries). Add Hypothesis-based tests.

### 30. Type annotations throughout `stats/tests.py`
`fisher_exact_test` and `glm_lrt_test` accept `MethylData` only via `TYPE_CHECKING` — the runtime signature is untyped. Add `from __future__ import annotations` and full runtime-compatible type hints.

---

## 📋 Priority Summary

| # | Priority | Category | Effort | RAM Impact |
|---|----------|----------|--------|------------|
| 1–4 | 🔴 P0 | Bugs | Low | Medium |
| 5–9 | 🟠 P1 | Quick RAM wins | Low–Medium | High |
| 10–14 | 🟡 P2 | Architecture | High | Very High |
| 15–20 | 🟢 P3 | New features | High | Low |
| 21–25 | 🔵 P4 | I/O | Medium | Medium |
| 26–30 | ⚙️ P5 | Quality | Medium | None |

**Recommended first sprint:** fix bugs 1–4, then tackle P1 items 5, 7, 8 — these are isolated changes that together can cut peak RAM by 40–60% with no architectural disruption.

**Recommended second sprint:** implement P2 item 10 (direct-to-Zarr streaming) and item 11 (chromosome-by-chromosome processing) — these are the two changes that actually eliminate the fundamental memory bottleneck, bringing epykit in line with how methylKit's `*DB` objects and dnmtools's streaming merge work.
