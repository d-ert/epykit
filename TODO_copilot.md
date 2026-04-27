### Findings

- `src/epykit/io/__init__.py` does **not** export DuckDB streaming functions yet (`build_anndata_streaming`, `read_samples_streaming`), so integration is incomplete at public API level.
- `read_samples()` in `src/epykit/io/sample_sheet.py` still routes to in-memory `build_anndata`, not the DuckDB path.
- `src/epykit/io/anndata_builder_duckdb.py` exists and is functional, but has performance/memory risks.

### Main risks in current DuckDB builder

1. **Large per-sample fetch spike**  
   `fetchnumpy()` materializes full joined columns for all loci per sample, causing a high transient allocation.

2. **Potential duplicate loci ambiguity**  
   Locus index uses `(chr, start)` only. If inputs ever contain duplicate starts or strand/context distinctions, joins may inflate rows or overwrite semantics.

3. **Hardcoded threading**  
   `SET threads TO 4` is static and may underuse or oversubscribe machines.

4. **No streaming/chunk fetch loop**  
   Full-column fetch prevents tighter memory control.

5. **Public API/test coverage gap**  
   No export from `io.__init__` and no dedicated tests for streaming wrapper behavior.

### Proposed integration and optimization plan

1. **API integration**
   - Export `build_anndata_streaming` and `read_samples_streaming` from `src/epykit/io/__init__.py`.
   - Keep existing API stable; add streaming as opt-in path.

2. **`read_samples` integration option**
   - Add `engine: Literal["auto","inmemory","duckdb"] = "auto"` to `read_samples`.
   - Route to DuckDB when `engine="duckdb"` (or auto-threshold by sample count/site estimates later).

3. **Memory optimization in DuckDB path**
   - Replace one-shot `fetchnumpy()` with chunked retrieval strategy.
   - Add configurable `duckdb_threads` parameter instead of hardcoded `4`.
   - Keep `duckdb_memory_limit` exposed; validate format early.

4. **Data integrity hardening**
   - Enforce/validate locus uniqueness contract.
   - Document exact coordinate key assumptions (currently effectively `chr:start` from Bismark cov).

5. **Tests**
   - Add tests for:
     - streaming exports/imports,
     - `read_samples(..., engine="duckdb")`,
     - parity checks vs in-memory builder on small fixtures,
     - join mode behavior (`outer`/`inner`),
     - missing file and schema validation parity.

6. **Docs**
   - Add concise README section: when to use streaming, expected memory profile, and caveats.

Toggle to Act mode to implement these changes.