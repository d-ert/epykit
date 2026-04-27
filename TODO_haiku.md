I've completed my assessment of the codebase. Here's the current state and optimization plan:

## Current State Analysis

**DuckDB Implementation Status:**
- ✓ Fully implemented in `anndata_builder_duckdb.py` (287 lines)
- ✓ Provides `build_anndata_streaming()` and `read_samples_streaming()` functions
- ✗ **NOT exported in `__init__.py`** — functions exist but are completely hidden from package API
- ✗ **NOT integrated into the default workflow** — `read_samples()` still uses the old RAM-heavy approach
- ✗ **No tests for DuckDB builder** — no test coverage despite being a critical optimization
- ✗ **Not in optional dependencies** — duckdb is not listed in `pyproject.toml`, so users can't install it

**Memory Problem with Current Approach:**
The default `read_samples()` → `build_anndata()` flow:
1. Loads all 6 Bismark files into Polars DataFrames in parallel (~6-14 GB)
2. Performs outer join on all 42M loci simultaneously in RAM
3. Creates 3 dense matrices (beta, coverage, counts) = ~3 GB
4. **Total peak: 17-20 GB** ❌ (crashes on most workstations)

**DuckDB Streaming Advantage:**
- Reads gzip files natively through DuckDB's I/O layer (no Python buffer inflation)
- Computes locus union entirely out-of-core inside DuckDB
- Processes one sample at a time, filling pre-allocated arrays
- **Peak RAM: ~4.5-6 GB** ✓ (3 GB output + 1-1.3 GB per sample chunk + DuckDB buffers)

## Recommended Integration Plan

### Phase 1: Export & Documentation (Quick Win)
1. Add `build_anndata_streaming` and `read_samples_streaming` to `__init__.py` exports
2. Update `__init__.py` docstring to mention the streaming option for large datasets
3. Add duckdb to optional dependencies in `pyproject.toml` as `streaming` extra
4. Create basic tests for the DuckDB builder in `test_io.py`

### Phase 2: Intelligent API (Smart Default)
1. Modify `read_samples()` to automatically detect dataset size
   - If `n_samples * estimated_sites > threshold` (e.g., 2B), suggest/auto-switch to streaming
   - Or add explicit `use_streaming=True` parameter
2. Add memory estimation utility function
3. Update docstring to explain when streaming is recommended

### Phase 3: Advanced Optimizations (Additional Improvements)
1. **Sparse matrix support** — detect high-sparsity data (many low-coverage sites) and use CSR format for counts/coverage
2. **Chunked reading** — improve DuckDB query chunking for even lower peak RAM (currently materializes full JOIN result)
3. **Parallel sample processing** — use Polars LazyFrame + DuckDB in hybrid mode for further speedup
4. **Coverage filtering in SQL** — move min/max_coverage filtering into DuckDB query (avoid fetching filtered-out sites)
5. **Compression options** — add option to store output as Zarr/HDF5 for on-disk datasets

## Expected Outcome
After Phase 1 & 2, users can:
```python
# Old way (RAM-hungry, ~17-20 GB peak)
adata = read_samples("cohort.csv", min_coverage=10)

# New way (efficient, ~5 GB peak)
adata = read_samples("cohort.csv", min_coverage=10, use_streaming=True)
# Or auto-detected:
adata = read_samples("cohort.csv", min_coverage=10)  # Uses streaming for large cohorts
```

Would you like me to proceed with implementation? I'd recommend starting with **Phase 1** (export + tests) to make the existing excellent DuckDB implementation immediately usable, then move to Phase 2 (smart defaults) for seamless integration.




Excellent question. Let me compare DuckDB against the alternatives mentioned in your research:

## Comparative Analysis of Approaches

### **1. DuckDB (Current Implementation) ⭐ BEST FOR EPYKIT**
**Pros:**
- In-process OLAP engine — no server setup, single-machine focus
- Native gzip support — reads `.cov.gz` directly without Python decompression
- Out-of-core joins — handles 42M loci union entirely in its buffer pool
- SQL-based filtering — `WHERE coverage >= min_cov` happens in query layer, not Python
- Vectorized execution — fast columnar operations
- Minimal dependencies — just one pip install
- Perfect for your exact use case (6 samples, 42M loci, single machine)

**Cons:**
- Not designed for distributed/cloud workflows (but not needed here)
- Slightly steeper learning curve than Pandas/Polars

**Peak RAM for 6×42M dataset: ~4.5–6 GB** ✓

---

### **2. Zarr + Chunked Writing (Your Secondary Implementation)**
**What you tried:**
- Pre-allocate Zarr array on disk
- Write chunks as each sample is processed
- Read back in to build final AnnData

**Why it failed:**
- Zarr v3 API changes broke `DirectoryStore`
- Unnecessary disk round-trip: **write 3GB → read 3GB = 6GB extra I/O**
- Still requires holding output arrays in RAM + one sample chunk
- Not actually *lower* peak RAM, just slower

**Peak RAM: 4–5 GB** (similar to DuckDB but with I/O overhead)

---

### **3. HDF5 (Like bsseq/bigmelon)**
**How other packages do it:**
```python
# bsseq stores M/Cov assays in HDF5 with DelayedArray laziness
# bigmelon uses GDS (Genomic Data Structure) format
```

**Pros:**
- On-disk backed — true "cold storage" option
- Random access — query any sample's methylation at any locus efficiently
- Hierarchical structure — easy to store per-chromosome subsets
- Mature ecosystem (h5py, Dask integration)

**Cons:**
- Extra complexity for your workflow (you don't need on-disk randomness *during* construction)
- Slower than DuckDB for batch operations
- Requires h5py + setup overhead
- Better suited for *post-construction* analysis, not loading phase

**Peak RAM during construction: 5–7 GB** (same as DuckDB but slower)

**Use case:** If you later want to store 100+ samples permanently and query them repeatedly, *then* HDF5 becomes attractive.

---

### **4. Tabix-indexed BED (Like methylKit)**
**How methylKit does it:**
```
1. Convert all samples to bgzip-compressed BED files
2. Index with tabix (genomic range queries)
3. Keep data on disk; query in chunks via tabix
```

**Pros:**
- Extremely efficient random access
- Standard bioinformatics format
- Portable across pipelines

**Cons:**
- Requires preprocessing step (bgzip + tabix indexing)
- Slower for bulk operations (your outer join)
- Need separate file per sample + index maintenance
- More disk I/O overhead than DuckDB

**Peak RAM during construction: 3–4 GB** (lowest because it's streaming-only, never materializes full join)

**Tradeoff:** Lower peak RAM but **much slower** for the initial cohort build.

---

### **5. Parquet Partitioned by Chromosome (Cloud-Native Approach)**
**How modern tools do it (TileDB, Polars datasets):**
```
Store: long-format data (locus, sample_id, beta, cov, meth)
Partition: by chromosome + sample
Query: via Polars LazyFrame or DuckDB on Parquet files
```

**Pros:**
- Columnar, compressed storage
- Lazy evaluation — doesn't materialize full join
- Cloud-native (S3-compatible)
- Scales to thousands of samples

**Cons:**
- Overkill for 6 samples
- More complex data layout
- Requires more disk space (long format vs wide)

**Peak RAM: 2–3 GB** (best!) but construction is slower

---

## Recommendation for EpyKit

### **Primary Path: Stick with DuckDB** ✅
**Why:**
1. **Perfect fit** — designed for exactly your problem (in-process, single-machine, batch joins)
2. **Already implemented** — you just need to expose it
3. **Peak RAM: 4.5–6 GB** — acceptable for most modern workstations
4. **Speed** — fastest among all options for construction
5. **Simplicity** — minimal dependencies, no file format gymnastics

### **Secondary Path: HDF5 for Optional On-Disk Storage**
After DuckDB constructs the AnnData, users can:
```python
# Build in RAM (fast)
adata = read_samples("cohort.csv", use_streaming=True)

# Optionally persist to HDF5 (for future repeated queries)
adata.write_h5ad("cohort.h5ad", compression="lzf")  # Already in AnnData
```

This is **not** for construction — just for users who want cold storage.

### **Hybrid Option for Future Scaling (100+ samples)**
If you ever need to scale beyond 6–50 samples:
- Keep DuckDB for building the locus index (out-of-core)
- Use Parquet partitioned dataset for on-disk storage
- Query with Polars LazyFrame (streaming)

But that's premature optimization for now.

---

## Bottom Line

| Approach | Peak RAM | Speed | Complexity | Best For |
|----------|----------|-------|------------|----------|
| **DuckDB** (current) | 4.5–6 GB | ⚡⚡⚡ Fast | Low | **Your case (6 samples)** |
| Zarr (broken) | 4–5 GB | Slow | Medium | — |
| HDF5 | 5–7 GB | ⚡ Medium | Low–Med | Post-construction storage |
| Tabix-indexed BED | 3–4 GB | ⚡ Slow | High | Random access queries |
| Parquet (cloud) | 2–3 GB | ⚡ Slow | High | Massive cohorts (100+) |

**My advice:** DuckDB is the right choice. The existing implementation is already excellent — it just needs to be:
1. **Exported** in the public API
2. **Documented** as the recommended path for large cohorts
3. **Tested** to ensure reliability
4. **Integrated** with smart auto-detection

Should I proceed with Phase 1 implementation to expose this to users?