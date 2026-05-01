## Plan: Reduce RAM usage and support disk-backed AnnData

TL;DR - Keep AnnData compatibility but switch large-cohort construction to on-disk, chunked stores (Zarr) and avoid materializing genome-scale arrays or string indices in memory. Update the DuckDB streaming builder to write into Zarr (chunked) rather than pre-allocating full NumPy arrays, and add an option in the chunked builder to return a Zarr-backed AnnData instead of loading arrays into RAM.

**Steps**
1. Update `build_anndata_chunked` to optionally return a Zarr-backed AnnData (new param `backed: bool = True` or `load_in_memory: bool = False`). Implement by: create Zarr arrays as currently done, and when `backed` is True, do not slice `X_zarr[:]` into memory; instead finish metadata (var/obs), then call `ad.read_zarr(str(zarr_path))` (or construct AnnData in a backed form) and attach `var`/`obs` carefully to avoid string-materialization. *depends on step 2 for var handling decisions*

2. Avoid materializing var index strings for large `n_sites`. For large datasets (>~1e6), attach `var` via direct assignment to `adata._var` (like the DuckDB builder does) or persist `var` as a separate lightweight CSV/Zarr and keep `adata` backed. Ensure `var` chromosome column is categorical and `start` is numeric to minimize memory.

3. Change `build_anndata_streaming` (DuckDB variant) to support Zarr-backed output instead of pre-allocating full in-memory arrays:
   - Add parameters: `zarr_path: PathLike | None = None`, `chunk_size: int = 1_000_000`, `backed: bool = True`, and `cleanup: bool = False`.
   - When `zarr_path` is provided, create Zarr datasets with shapes (n_samples, n_sites) and appropriate chunking (chunks=(n_samples, min(chunk_size, n_sites))). Use `fill_value` as configured.
   - Replace `beta_mat`, `cov_mat`, `meth_mat` preallocation with writes into the Zarr arrays. For each sample, after obtaining `locus_idx` and column values from DuckDB, write `zarr_X[i, locus_idx] = beta_vals` (and similarly for layers). This keeps peak RAM to one sample JOIN result plus small buffers.
   - For very sparse datasets, optionally write compressed sparse rows to a disk-backed sparse representation (advanced; can be postponed).
   - After filling all rows, either return a backed AnnData via `ad.read_zarr(zarr_path)` or (if `backed=False`) load slices into memory as before.

4. Var streaming and categories:
   - Keep the current DuckDB approach of streaming chr codes and start arrays in chunks into NumPy arrays; however, if returning a backed Anndata, persist `var` metadata to a compact CSV or Zarr table and attach lazily. Avoid creating an object with millions of Python strings.
   - If attaching `var` to an in-memory AnnData is required, warn and/or refuse for very large `n_sites` (user must call with `backed=False` explicitly).

5. API and usability:
   - Add clear parameters and documentation: `zarr_path`, `backed`, `chunk_size`, `cleanup`. Document recommended defaults for large genomes.
   - Update `src/epykit/io/__init__.py` exports and README examples to show how to create a backed AnnData and how to load it lazily.

6. Tests and verification:
   - Add unit tests that build small toy cohorts with `backed=True` and `backed=False` and verify shapes and values.
   - Add an integration memory-test that runs on synthetic larger locus counts (e.g., 5M sites in CI-lite mode) to assert peak RSS < target (configurable). For full-scale (~42M), provide benchmark script only (too large for CI).
   - Run existing test suite and a new benchmark demonstrating memory usage with `psutil` instrumentation.

7. Docs and migration notes:
   - Update docs with migration notes: prefer `build_anndata_chunked(..., backed=True, zarr_path=...)` for >1M sites; explain tradeoffs (slower random access vs. low RAM) and how to convert backed AnnData to in-memory if needed.

**Relevant files**
- `/workspaces/epykit/src/epykit/io/anndata_builder_duckdb.py` — modify to support writing per-sample results into Zarr arrays instead of pre-allocating NumPy arrays; add `zarr_path`, `chunk_size`, `backed`, `cleanup` params and var persistence changes.
- `/workspaces/epykit/src/epykit/io/anndata_builder_chunked.py` — change final assembly to optionally return a Zarr-backed AnnData (do not load `X_zarr[:]` into memory when `backed=True`) and add parameters.
- `/workspaces/epykit/src/epykit/io/anndata_builder.py` — update docs and possibly expose a helper `to_zarr_backed(adata, path)`.
- `/workspaces/epykit/src/epykit/io/__init__.py` — export updated functions and document new signatures.
- `tests/` — add tests for backed chunked/streaming builders and a small memory/behavior test.

**Verification**
1. Build a small synthetic dataset (e.g., 10 samples × 5M loci) with `backed=True` and measure RSS during construction; ensure RSS stays under ~2–3 GB depending on DuckDB pool config.
2. Verify that `ad = build_anndata_chunked(..., backed=True)` returns an object that can be used by downstream code for reading X, layers (they should load on-demand from disk).
3. For DuckDB streaming: run full pipeline on a 42M-site test in a machine with constrained RAM and confirm construction completes when using Zarr-backed mode.
4. Run unit tests to ensure API parity and no regressions.

**Decisions / Assumptions**
- Keep AnnData API compatibility; do not remove AnnData as the primary format — instead prefer returning an on-disk (Zarr) backed AnnData for large datasets.
- Use Zarr as the main on-disk storage because it supports chunked writes and is already present in the repo (`anndata_builder_chunked`), and AnnData integrates with Zarr.
- Sparse on-disk representations are more complex; implement dense Zarr first, then evaluate sparse-disk options if needed.

**Further Considerations**
1. Do you want `build_anndata_streaming` to default to `backed=True` for large datasets, or require the caller to opt-in? Recommendation: require explicit `zarr_path` or `backed=True` to avoid surprising memory behavior.
2. For user ergonomics, provide helper `open_backed(zarr_path)` that returns a convenient AnnData view and documents how to compute downstream analyses without loading everything.
3. Investigate whether `anndata` supports constructing an `AnnData` that references Zarr arrays directly (without `ad.read_zarr`) — if stable, that can avoid a second pass and make var handling simpler.

