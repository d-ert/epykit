## What I found (current state)

1) **DuckDB builder exists but is not integrated**
- `src/epykit/io/anndata_builder_duckdb.py` implements `build_anndata_streaming()` + `read_samples_streaming()`.
- It is **not exported** from `epykit.io` (`src/epykit/io/__init__.py`), not referenced by `read_samples()`, and has **no tests**.

2) **Current default pipeline is guaranteed to OOM for your 42M-locus union**
- `read_samples()` loads all samples into Polars DataFrames in parallel, then `build_anndata()` does union/join in RAM.
- `build_anndata_chunked()` still reads all sample frames into RAM and also has a very expensive `is_in(chunk_locus_ids)` filter (will be painfully slow at 42M).
- `build_anndata_chunked()` finally does `X_zarr[:]` which **loads everything back into RAM**, so it’s not truly disk-backed.

3) **Important hidden memory killer: `adata.var` for 42M sites**
Even if we solve matrix construction, a 42M-row pandas `var` with object/string columns can easily be multiple GB. We must:
- avoid generating per-site Python strings (like `"chr1:123-124:*"`) as the index
- keep `var` minimal and use memory-efficient dtypes (categoricals + int32)

---

## Design goals (based on your decisions)

You said:
- **duckdb required dependency**
- `read_samples()` should get an `engine=` parameter
- prefer **disk-backed output** (and possibly sparse later)

So the target is:
- A single public entry point: `epykit.io.read_samples(..., engine='duckdb', output='zarr')`
- Peak RAM dominated by *buffers* only (not full matrices)
- A persistent Zarr store that you can re-open later and continue analysis

---

## Proposed plan (what I will implement next in ACT mode)

### 1) Make DuckDB a first-class, required backend
**Files**
- `pyproject.toml`: add `duckdb` to `[project].dependencies`.

**Success criteria**
- `import duckdb` works in normal installs.

### 2) Integrate DuckDB into the normal workflow (no separate function needed)
**Change** `src/epykit/io/sample_sheet.py::read_samples()`
- Add parameter:
  ```python
  def read_samples(..., engine: str = 'polars', output: str = 'memory', out_path: str|Path|None = None, ...)
  ```
- Behavior:
  - `engine='polars'` → current behavior
  - `engine='duckdb'` → do **not** load Polars DataFrames; instead parse the sample sheet and call a DuckDB-based builder.

**Also update** `src/epykit/io/__init__.py` to export the new capabilities.

**Success criteria**
- Existing code keeps working unchanged.
- New call works:
  ```python
  adata = epykit.io.read_samples('cohort.csv', engine='duckdb', output='zarr', out_path='cohort.zarr')
  ```

### 3) Implement a real disk-backed DuckDB builder (stream into Zarr)
Right now `build_anndata_streaming()` pre-allocates dense NumPy arrays → still several GB in RAM. For disk-backed, we’ll implement:
- `build_anndata_duckdb_zarr(...)` (name can vary) that:
  1. builds `_loci` index inside DuckDB
  2. creates Zarr arrays on disk for `X`, `layers/coverage`, `layers/methylated_counts`
  3. for each sample, executes the LEFT JOIN and **fetches results in batches** (Arrow record batches) and writes directly to the Zarr arrays.

Key implementation details:
- Use SQL `COALESCE` so we never materialize masked integer arrays:
  - coverage/methylated missing → 0
  - beta missing → NaN
- Store site index as **int64 locus_id** (chr_code * scale + start) in `adata.var_names`, not `locus_key` strings.
- Create `adata.var` with memory-efficient dtypes:
  - `chr`: `pd.Categorical`
  - `start`, `end`: `int32` (safe for mammalian genomes)
  - `strand`/`context`: optional or categorical constants

**Success criteria**
- Building 6×42M no longer spikes to 15–20GB RAM.
- Output is a `.zarr` directory that can be reopened via `epykit.io.load()` / `ad.read_zarr()`.

### 4) Fix the “var_names == locus_key string” assumption across the codebase
Right now, interval code assumes the index column created by `reset_index()` is named `locus_key` (see `intervals/tiling.py`). But `build_anndata()`/`build_anndata_chunked()` currently set index name to `locus_id`.

I’ll make the interval helpers robust by:
- not hardcoding `"locus_key"` during `reset_index()`
- using `adata.var.index.name` (or fallback to `'index'`) to rename to `_site_key`

Separately, stats output currently renames var_names to a column called `locus_key`. We can still output a `locus_key` column by constructing it from `chr/start/end/strand` on the fly (cheap relative to 42M *during analysis* only after filtering/unite/tiling).

**Success criteria**
- `tile_counts()`, `annotate_features()`, `annotate_cpg_islands()` work regardless of whether `var_names` are strings or integers.

### 5) Make chunked builder actually disk-backed (quick win)
`build_anndata_chunked()` currently writes Zarr but then does `X_zarr[:]` and returns a fully in-memory AnnData.

I’ll change it to either:
- return `ad.read_zarr(zarr_path)` (true disk-backed), or
- offer `output='memory'|'zarr'` like the DuckDB engine.

**Success criteria**
- `build_anndata_chunked(..., output='zarr')` does not reload into RAM.

### 6) Tests + documentation
**Tests** (`tests/test_io.py`)
- Add one integration test:
  - create a small sample sheet with the toy `.cov` files
  - run `read_samples(..., engine='duckdb', output='memory')` to validate correctness quickly
  - optionally test `output='zarr'` writes a store and can be reopened

**Docs**
- Update `README.md` “Quick Start” section to recommend `engine='duckdb'` for large cohorts.

---

## Notes about “sparse CSR”
Sparse CSR and **NaN for missing sites** don’t mix well (sparse matrices typically treat missing as 0). For methylation, missing coverage is semantically different from 0% methylation.

Best practical approach:
- keep disk-backed dense Zarr with NaNs (correct semantics)
- after you subset/unite/filter down to a manageable size, optionally provide a helper to convert `X` to sparse (replacing NaN→0 plus storing a mask if needed)

If you still want “sparse-first”, we can do it, but we’ll need an explicit missingness representation strategy.

---